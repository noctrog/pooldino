"""Train an independent ViT-XL segmentation decoder on frozen PoolDINO tokens.

Example::

    python -m pooldino.train_segmentation \
      --source-checkpoint \
        output/decoders/repeatconv2x2-dinol-vitxl-raev2official-tfds \
      --experiment repeatconv2x2-ade20k-s42 --use-wandb

The source DINO and EMA RepeatConv are frozen.  Compressed tokens are nearest
repeated to 16x16 before receiving distinct positional embeddings in the new
task ViT.  Only the new ViT-XL and 150-class output projection are optimized.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
import wandb
from absl import logging

from pooldino.pooled_dense_decoder import (
    DenseOptimConfig,
    PooledTaskDecoder,
    PooledTaskDecoderConfig,
    configure_task_attention,
    encode_repeated_pooled_tokens,
    initialize_task_vit_from_rgb_decoder,
    restore_frozen_pooled_source,
    source_encoding_kwargs,
    task_decoder_config_from_source,
)
from pooldino.pooled_generator import PooledDecoderIdentity
from pooldino.segmentation.configs import SegDataConfig
from pooldino.segmentation.data import create_seg_dataloaders
from pooldino.segmentation.miou import ConfusionMatrix
from pooldino.training import (
    BaseArgs,
    build_lr_schedule,
    build_optimizer,
    create_checkpoint_manager,
    init_wandb,
    jax_system_metrics,
    save_checkpoint,
    setup_context,
    setup_progress,
)
from pooldino.utils import (
    determine_save_path,
    open_restore_manager,
    prefetch_to_mesh,
    restore_data_loader,
    restore_optimizer_state,
)


SEGMENTATION_ITEM_NAMES = ["decoder", "optim", "loader", "config"]


@dataclass
class PooledSegmentationConfig:
    train: DenseOptimConfig
    data: SegDataConfig
    decoder: PooledTaskDecoderConfig
    source_identity: PooledDecoderIdentity
    source_pool_window: tuple[int, int]
    source_unique_grid: tuple[int, int]
    vit_initialization: PooledDecoderIdentity | None = None
    task: Literal["ade20k_semantic_official"] = "ade20k_semantic_official"


@dataclass
class Args(BaseArgs):
    source_checkpoint: Path = field(
        default_factory=lambda: Path(
            "output/decoders/"
            "repeatconv2x2-dinol-vitxl-raev2official-tfds"
        )
    )
    """Pooled RGB checkpoint root. Only config and tokenizer_ema are restored."""
    source_step: int = 40032
    dinov3_checkpoint_path: Path | None = None
    implementation: Literal["cudnn", "xla"] = "xla"
    vit_init_checkpoint: Path | None = None
    """Optional pooled RGB decoder whose ViT parameters initialize this task ViT."""
    vit_init_step: int = 40032
    vit_init_use_ema: bool = True

    project_name: str = "pooldino-ade20k"
    experiment: str = "repeatconv2x2-ade20k-s42"
    gpu_batch_size: int = 2
    val_epochs_freq: int = 5

    epochs: int = 80
    global_batch_size: int = 32
    lr_peak: float = 2e-4
    lr_final: float = 1e-6
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    grad_clip_norm: float = 3.0

    mask_resolution: int = 512
    ade20k_root: Path = Path(".data/ADEChallengeData2016")
    """Official ADEChallengeData2016 root containing images/ and annotations/."""
    color_jitter: bool = True
    val_keep_ratio: bool = False
    """Keep false for aspect-preserving, padded, batchable 512px validation."""


def _masked_ce_statistics(
    logits: jax.Array,
    masks: jax.Array,
    *,
    num_classes: int,
    ignore_index: int,
    valid_batch_size: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    valid = masks != ignore_index
    if valid_batch_size is not None:
        sample_valid = jnp.arange(masks.shape[0]) < valid_batch_size
        valid = valid & sample_valid[:, None, None]
    flat_logits = logits.reshape(-1, num_classes).astype(jnp.float32)
    flat_masks = masks.reshape(-1)
    valid = valid.reshape(-1)
    safe_target = jnp.where(valid, flat_masks, 0)
    per_pixel = optax.softmax_cross_entropy_with_integer_labels(
        flat_logits, safe_target
    )
    loss_sum = jnp.sum(jnp.where(valid, per_pixel, 0.0))
    valid_count = jnp.sum(valid, dtype=jnp.float32)
    predictions = jnp.argmax(flat_logits, axis=-1)
    correct = jnp.sum(valid & (predictions == safe_target), dtype=jnp.float32)
    loss = loss_sum / jnp.maximum(valid_count, 1.0)
    return loss, loss_sum, valid_count, correct


_SOURCE_STATIC_ARGS = (
    "backbone_resolution",
    "num_prefix_tokens",
    "layer_indices",
    "aggregation",
    "normalize_each",
    "add_final_mean",
    "post_pool_norm",
    "representation_eps",
    "grid_hw",
    "pool_hw",
)


@nnx.jit(
    donate_argnames=("decoder", "optim"),
    static_argnames=(
        "output_hw",
        "num_classes",
        "ignore_index",
        *_SOURCE_STATIC_ARGS,
    ),
)
def segmentation_train_step(
    dino,
    tokenizer,
    decoder: PooledTaskDecoder,
    optim: nnx.Optimizer,
    images: jax.Array,
    masks: jax.Array,
    *,
    output_hw: tuple[int, int],
    num_classes: int,
    ignore_index: int,
    backbone_resolution: int,
    num_prefix_tokens: int,
    layer_indices: tuple[int, ...],
    aggregation: str,
    normalize_each: bool,
    add_final_mean: bool,
    post_pool_norm: bool,
    representation_eps: float,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> dict[str, jax.Array]:
    def loss_fn(model: PooledTaskDecoder):
        tokens = encode_repeated_pooled_tokens(
            images,
            dino,
            tokenizer,
            backbone_resolution=backbone_resolution,
            num_prefix_tokens=num_prefix_tokens,
            layer_indices=layer_indices,
            aggregation=aggregation,
            normalize_each=normalize_each,
            add_final_mean=add_final_mean,
            post_pool_norm=post_pool_norm,
            representation_eps=representation_eps,
            grid_hw=grid_hw,
            pool_hw=pool_hw,
        )
        logits = model(tokens, output_hw=output_hw, deterministic=False)
        loss, _, valid_count, correct = _masked_ce_statistics(
            logits,
            masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        metrics = {
            "loss": loss,
            "pixel_acc": correct / jnp.maximum(valid_count, 1.0),
            "valid_pixels": valid_count,
        }
        return loss, metrics

    (_, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(decoder)
    optim.update(decoder, gradients)
    return metrics


@nnx.jit(
    static_argnames=(
        "output_hw",
        "num_classes",
        "ignore_index",
        *_SOURCE_STATIC_ARGS,
    )
)
def segmentation_eval_step(
    dino,
    tokenizer,
    decoder: PooledTaskDecoder,
    images: jax.Array,
    masks: jax.Array,
    valid_batch_size: jax.Array,
    *,
    output_hw: tuple[int, int],
    num_classes: int,
    ignore_index: int,
    backbone_resolution: int,
    num_prefix_tokens: int,
    layer_indices: tuple[int, ...],
    aggregation: str,
    normalize_each: bool,
    add_final_mean: bool,
    post_pool_norm: bool,
    representation_eps: float,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens = encode_repeated_pooled_tokens(
        images,
        dino,
        tokenizer,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        post_pool_norm=post_pool_norm,
        representation_eps=representation_eps,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
    )
    logits = decoder(tokens, output_hw=output_hw, deterministic=True)
    _, loss_sum, valid_count, _ = _masked_ce_statistics(
        logits,
        masks,
        num_classes=num_classes,
        ignore_index=ignore_index,
        valid_batch_size=valid_batch_size,
    )
    predictions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return predictions, loss_sum, valid_count


def run_validation(
    restored,
    decoder: PooledTaskDecoder,
    val_iter,
    *,
    mesh,
    prefetch: int,
    cfg: PooledSegmentationConfig,
) -> dict[str, float | list[float]]:
    decoder.eval()
    confusion = ConfusionMatrix(cfg.data.num_classes)
    total_loss = 0.0
    total_valid = 0.0
    static = {
        **source_encoding_kwargs(restored),
        "output_hw": (cfg.data.mask_resolution, cfg.data.mask_resolution),
        "num_classes": cfg.data.num_classes,
        "ignore_index": cfg.data.ignore_index,
    }
    for batch in prefetch_to_mesh(
        val_iter,
        prefetch,
        mesh,
        pad_to=mesh.size,
    ):
        predictions, loss_sum, valid_count = segmentation_eval_step(
            restored.dino,
            restored.tokenizer,
            decoder,
            batch["image"],
            batch["mask"],
            batch["_valid_size"],
            **static,
        )
        valid_size = int(np.asarray(jax.device_get(batch["_valid_size"])))
        predictions_host = np.asarray(predictions)[:valid_size]
        masks_host = np.asarray(batch["mask"])[:valid_size]
        confusion.update(
            predictions_host,
            masks_host,
            ignore_index=cfg.data.ignore_index,
        )
        total_loss += float(loss_sum)
        total_valid += float(valid_count)
    summary = confusion.summarize()
    summary["loss"] = total_loss / max(total_valid, 1.0)
    summary["valid_pixels"] = total_valid
    decoder.train()
    return summary


def _validate_resume_source(
    manager: ocp.CheckpointManager,
    step: int,
    identity: PooledDecoderIdentity,
    *,
    vit_init_checkpoint: Path | None,
    vit_init_step: int,
    vit_init_use_ema: bool,
) -> PooledDecoderIdentity | None:
    saved = manager.restore(
        step,
        args=ocp.args.Composite(config=ocp.args.JsonRestore()),
    )["config"]
    source = saved.get("source_identity", {})
    for name in ("checkpoint_step", "use_ema", "stage1_profile", "config_sha256"):
        if source.get(name) != getattr(identity, name):
            raise ValueError(
                f"Task checkpoint source mismatch for {name}: "
                f"saved={source.get(name)!r}, requested={getattr(identity, name)!r}."
            )
    saved_initialization = saved.get("vit_initialization")
    initialization = (
        None
        if saved_initialization is None
        else PooledDecoderIdentity(**saved_initialization)
    )
    if vit_init_checkpoint is None:
        return initialization
    if initialization is None:
        raise ValueError(
            "The task checkpoint was trained from scratch, but a ViT initializer "
            "was requested while resuming it."
        )
    requested_path = str(vit_init_checkpoint)
    if "://" not in requested_path:
        requested_path = str(vit_init_checkpoint.expanduser().resolve())
    for name, requested in (
        ("checkpoint_path", requested_path.rstrip("/")),
        ("checkpoint_step", vit_init_step),
        ("use_ema", vit_init_use_ema),
    ):
        if getattr(initialization, name) != requested:
            raise ValueError(
                f"Task checkpoint ViT initialization mismatch for {name}: "
                f"saved={getattr(initialization, name)!r}, requested={requested!r}."
            )
    return initialization


def main(args: Args) -> dict[str, float | list[float]]:
    if args.val_keep_ratio:
        raise ValueError(
            "--val-keep-ratio is not supported by the primary fixed-lattice "
            "protocol; use aspect-preserving fixed-size validation padding."
        )
    ctx = setup_context(args)
    restored = restore_frozen_pooled_source(
        args.source_checkpoint,
        mesh=ctx.mesh,
        mp=ctx.mp_policy,
        step=args.source_step,
        implementation=args.implementation,
        dinov3_checkpoint_path=args.dinov3_checkpoint_path,
    )
    decoder_cfg = task_decoder_config_from_source(
        restored,
        output_channels=150,
    )
    train_cfg = DenseOptimConfig(
        epochs=args.epochs,
        batch_size=args.global_batch_size,
        lr_peak=args.lr_peak,
        lr_final=args.lr_final,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
    )
    source_mean = tuple(restored.data_cfg.normalization_mean)
    source_std = tuple(restored.data_cfg.normalization_std)
    data_cfg = SegDataConfig(
        dataset_key="ade",
        tfds_name="ade20k_semantic",
        train_split="train",
        val_split="validation",
        mask_field="annotation",
        num_classes=150,
        ignore_index=-1,
        raw_ignore_index=0,
        mask_resolution=args.mask_resolution,
        encoder_resolution=restored.cfg.backbone_resolution,
        rrc_scale=(0.5, 2.0),
        color_jitter=args.color_jitter,
        data_dir=str(args.ade20k_root.expanduser().resolve()),
        normalization_mean=source_mean,
        normalization_std=source_std,
        val_keep_ratio=args.val_keep_ratio,
        patch_size=16,
        num_workers=(
            args.num_data_workers if args.num_data_workers is not None else 8
        ),
    )
    micro_batch = args.gpu_batch_size * ctx.data_parallel_size
    if train_cfg.batch_size % micro_batch != 0:
        raise ValueError(
            f"global batch {train_cfg.batch_size} must be divisible by device "
            f"micro batch {micro_batch}."
        )
    grad_acc_steps = train_cfg.batch_size // micro_batch
    data = create_seg_dataloaders(
        data_cfg,
        batch_size=micro_batch,
        train_epochs=None,
        val_epochs=1,
        num_workers=data_cfg.num_workers,
        gcs_bucket=args.gcs_bucket if args.data_in_bucket else None,
    )
    total_updates = (data.train_ds_size * train_cfg.epochs) // train_cfg.batch_size
    steps_per_epoch = max(data.train_ds_size // train_cfg.batch_size, 1)
    val_interval = (
        total_updates
        if args.val_epochs_freq <= 0
        else max(steps_per_epoch * args.val_epochs_freq, 1)
    )
    lr_schedule = build_lr_schedule(train_cfg, total_updates)

    default_path = f"output/segmentation/{args.experiment}"
    restore_manager, restore_step = open_restore_manager(
        args.restore,
        args.maybe_restore,
        default_path=default_path,
        gcs_bucket=args.gcs_bucket,
        item_names=SEGMENTATION_ITEM_NAMES,
        restore_best=args.restore_best,
        best_metric_key="miou",
        best_mode="max",
    )
    if restore_manager is None:
        decoder = PooledTaskDecoder(decoder_cfg, ctx.mp_policy, rngs=ctx.rngs)
        vit_initialization = None
        if args.vit_init_checkpoint is not None:
            vit_initialization = initialize_task_vit_from_rgb_decoder(
                decoder,
                args.vit_init_checkpoint,
                mesh=ctx.mesh,
                mp=ctx.mp_policy,
                step=args.vit_init_step,
                use_ema=args.vit_init_use_ema,
            )
    else:
        vit_initialization = _validate_resume_source(
            restore_manager,
            restore_step,
            restored.identity,
            vit_init_checkpoint=args.vit_init_checkpoint,
            vit_init_step=args.vit_init_step,
            vit_init_use_ema=args.vit_init_use_ema,
        )
        decoder = PooledTaskDecoder.restore(
            restore_manager,
            restore_step,
            "decoder",
            ctx.mesh,
            decoder_cfg,
            ctx.mp_policy,
        )
    cfg = PooledSegmentationConfig(
        train=train_cfg,
        data=data_cfg,
        decoder=decoder_cfg,
        source_identity=restored.identity,
        source_pool_window=restored.cfg.pool_window,
        source_unique_grid=restored.latent_spec.grid,
        vit_initialization=vit_initialization,
    )
    configure_task_attention(decoder, args.implementation)
    optimizer = build_optimizer(
        decoder,
        nnx.state(decoder, nnx.Param),
        lr_schedule,
        adam_b1=train_cfg.adam_b1,
        adam_b2=train_cfg.adam_b2,
        weight_decay=train_cfg.weight_decay,
        grad_clip_norm=train_cfg.grad_clip_norm,
        grad_acc_steps=grad_acc_steps,
        wd_exclude_names={"pos_embed", "cls_token", "bias"},
    )
    data_iter = iter(data.train_loader)
    if restore_manager is not None:
        restore_optimizer_state(
            restore_manager, restore_step, optimizer, ctx.mesh
        )
        data_iter = restore_data_loader(
            restore_manager, restore_step, data_iter
        )

    save_path = determine_save_path(
        args.checkpoint,
        args.checkpoint_dir,
        default_path,
        args.gcs_bucket,
    )
    checkpoint_manager = create_checkpoint_manager(
        save_path,
        SEGMENTATION_ITEM_NAMES,
        save_interval_steps=val_interval,
        total_steps=total_updates,
        best_fn=lambda metrics: metrics["miou"],
        best_mode="max",
        # Preserve the latest checkpoint (including the forced final save) and
        # the best-validation checkpoint. This also gives queue launchers a
        # durable final-step completion artifact.
        max_to_keep=1,
        best_n=1,
        keep_without_metrics=args.keep_checkpoints_without_metrics,
    )
    use_wandb, wandb_resume_step = init_wandb(
        args,
        cfg,
        project_name=args.project_name,
        group="pooldino-dense-tasks",
    )

    parameter_count = sum(
        int(value.size) for value in jax.tree.leaves(nnx.state(decoder, nnx.Param))
    )
    logging.info(
        "ADE20K source=%s unique_grid=%s pool=%s decoder_params=%.2fM "
        "updates=%d grad_acc=%d vit_init=%s",
        args.source_checkpoint,
        restored.latent_spec.grid,
        restored.cfg.pool_window,
        parameter_count / 1e6,
        total_updates,
        grad_acc_steps,
        (
            vit_initialization.checkpoint_path
            if vit_initialization is not None
            else "scratch"
        ),
    )

    progress, profiler, updates_completed = setup_progress(
        args, optimizer, total_updates
    )
    train_step_cached = nnx.cached_partial(
        segmentation_train_step,
        restored.dino,
        restored.tokenizer,
        decoder,
        optimizer,
    )
    static = {
        **source_encoding_kwargs(restored),
        "output_hw": (data_cfg.mask_resolution, data_cfg.mask_resolution),
        "num_classes": data_cfg.num_classes,
        "ignore_index": data_cfg.ignore_index,
    }
    micro_step = updates_completed * grad_acc_steps
    final_summary: dict[str, float | list[float]] | None = None
    best_miou = (
        float(wandb.run.summary.get("best_val_miou", float("-inf")))
        if use_wandb and wandb.run is not None
        else float("-inf")
    )
    start_time = time.perf_counter()

    for batch in prefetch_to_mesh(data_iter, args.prefetch, ctx.mesh):
        if updates_completed >= total_updates:
            break
        micro_step += 1
        metrics = train_step_cached(batch["image"], batch["mask"], **static)
        if micro_step % grad_acc_steps != 0:
            continue

        updates_completed += 1
        progress.update(1)
        profiler.step(updates_completed)
        if (
            use_wandb
            and updates_completed > wandb_resume_step
            and updates_completed % args.wandb_log_every == 0
        ):
            wandb.log(
                {
                    "train/loss": float(metrics["loss"]),
                    "train/pixel_acc": float(metrics["pixel_acc"]),
                    "train/valid_pixels": float(metrics["valid_pixels"]),
                    "train/lr": float(lr_schedule(updates_completed)),
                    "train/epoch": updates_completed / steps_per_epoch,
                    "step": updates_completed,
                    **jax_system_metrics(),
                }
            )

        run_val = (
            updates_completed % val_interval == 0
            or updates_completed >= total_updates
        )
        if run_val:
            final_summary = run_validation(
                restored,
                decoder,
                iter(data.val_loader),
                mesh=ctx.mesh,
                prefetch=args.prefetch,
                cfg=cfg,
            )
            best_miou = max(best_miou, float(final_summary["miou"]))
            logging.info(
                "step=%d val_loss=%.5f mIoU=%.4f mAcc=%.4f pixel_acc=%.4f",
                updates_completed,
                final_summary["loss"],
                final_summary["miou"],
                final_summary["macc"],
                final_summary["pixel_acc"],
            )
            if use_wandb and updates_completed > wandb_resume_step:
                wandb.log(
                    {
                        "val/loss": final_summary["loss"],
                        "val/miou": final_summary["miou"],
                        "val/best_miou": best_miou,
                        "val/macc": final_summary["macc"],
                        "val/pixel_acc": final_summary["pixel_acc"],
                        "val/valid_pixels": final_summary["valid_pixels"],
                        "step": updates_completed,
                    }
                )
                wandb.run.summary["best_val_miou"] = best_miou
            save_checkpoint(
                checkpoint_manager,
                updates_completed,
                decoder,
                optimizer,
                data_iter,
                cfg,
                metrics={
                    "miou": float(final_summary["miou"]),
                    "loss": float(final_summary["loss"]),
                },
            )

        if updates_completed >= total_updates:
            break

    progress.close()
    logging.info("Training completed in %.1f seconds.", time.perf_counter() - start_time)
    if final_summary is None:
        final_summary = run_validation(
            restored,
            decoder,
            iter(data.val_loader),
            mesh=ctx.mesh,
            prefetch=args.prefetch,
            cfg=cfg,
        )
    if checkpoint_manager is not None:
        checkpoint_manager.wait_until_finished()
    if restore_manager is not None:
        restore_manager.close()
    if use_wandb:
        wandb.finish()
    return final_summary


if __name__ == "__main__":
    main(tyro.cli(Args))
