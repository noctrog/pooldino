"""Train an independent ViT-XL metric-depth decoder on frozen PoolDINO tokens.

Example::

    python -m pooldino.train_depth \
      --source-checkpoint \
        output/decoders/repeatconv2x2-dinol-vitxl-raev2official-tfds \
      --experiment repeatconv2x2-nyuv2-s42 --use-wandb

This uses the TFDS ``nyu_depth_v2`` expanded training split and its 654-image
validation split.  The protocol identity, depth bounds, and evaluation crop
are persisted in every task checkpoint.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
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
from pooldino.pooled_depth import (
    DepthDataConfig,
    create_depth_dataloaders,
    depth_sufficient_statistics,
    depth_valid_mask,
    log_depth_gradient_loss,
    resize_log_depth_in_metric_space,
    silog_loss,
    summarize_depth_statistics,
)
from pooldino.pooled_generator import PooledDecoderIdentity
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


DEPTH_ITEM_NAMES = ["decoder", "optim", "loader", "config"]


@dataclass
class PooledDepthConfig:
    train: DenseOptimConfig
    data: DepthDataConfig
    decoder: PooledTaskDecoderConfig
    source_identity: PooledDecoderIdentity
    source_pool_window: tuple[int, int]
    source_unique_grid: tuple[int, int]
    vit_initialization: PooledDecoderIdentity | None = None
    task: Literal["nyuv2_metric_depth"] = "nyuv2_metric_depth"
    protocol: Literal["nyuv2_tfds_expanded"] = "nyuv2_tfds_expanded"


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

    project_name: str = "pooldino-nyuv2"
    experiment: str = "repeatconv2x2-nyuv2-s42"
    gpu_batch_size: int = 4
    val_epochs_freq: int = 1

    epochs: int = 50
    global_batch_size: int = 32
    lr_peak: float = 2e-4
    lr_final: float = 1e-6
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    grad_clip_norm: float = 3.0

    min_depth: float = 0.1
    max_depth: float = 10.0
    silog_lambda: float = 0.5
    gradient_weight: float = 0.0
    color_jitter: bool = True
    use_eval_crop: bool = True
    """Use the common NYUv2 crop [45:471, 41:601] for validation metrics."""


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


def _batch_depth_metrics(
    statistics: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    count = jnp.maximum(statistics["count"], 1.0)
    return {
        "abs_rel": statistics["abs_rel_sum"] / count,
        "rmse": jnp.sqrt(statistics["sq_error_sum"] / count),
        "log_rmse": jnp.sqrt(statistics["log_sq_error_sum"] / count),
        "delta1": statistics["delta1_count"] / count,
        "valid_pixels": statistics["count"],
    }


@nnx.jit(
    donate_argnames=("decoder", "optim"),
    static_argnames=(
        "output_hw",
        "min_depth",
        "max_depth",
        "silog_lambda",
        "gradient_weight",
        *_SOURCE_STATIC_ARGS,
    ),
)
def depth_train_step(
    dino,
    tokenizer,
    decoder: PooledTaskDecoder,
    optim: nnx.Optimizer,
    images: jax.Array,
    target_depth: jax.Array,
    *,
    output_hw: tuple[int, int],
    min_depth: float,
    max_depth: float,
    silog_lambda: float,
    gradient_weight: float,
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
        native_output_hw = (
            model.cfg.output_grid[0] * model.cfg.output_patch_size,
            model.cfg.output_grid[1] * model.cfg.output_patch_size,
        )
        native_log_prediction = model(
            tokens,
            output_hw=native_output_hw,
            deterministic=False,
        )[..., 0]
        log_prediction = resize_log_depth_in_metric_space(
            native_log_prediction,
            output_hw=output_hw,
        )
        valid = depth_valid_mask(
            target_depth,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        silog = silog_loss(
            log_prediction,
            target_depth,
            valid,
            coefficient=silog_lambda,
        )
        gradient, _, _ = log_depth_gradient_loss(
            log_prediction,
            target_depth,
            valid,
        )
        loss = silog + gradient_weight * gradient
        statistics = depth_sufficient_statistics(
            log_prediction,
            target_depth,
            valid,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        metrics = {
            "loss": loss,
            "silog": silog,
            "gradient_error": gradient,
            **_batch_depth_metrics(statistics),
        }
        return loss, metrics

    (_, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(decoder)
    optim.update(decoder, gradients)
    return metrics


@nnx.jit(
    static_argnames=(
        "output_hw",
        "min_depth",
        "max_depth",
        "eval_crop",
        *_SOURCE_STATIC_ARGS,
    )
)
def depth_eval_step(
    dino,
    tokenizer,
    decoder: PooledTaskDecoder,
    images: jax.Array,
    target_depth: jax.Array,
    valid_batch_size: jax.Array,
    *,
    output_hw: tuple[int, int],
    min_depth: float,
    max_depth: float,
    eval_crop: tuple[int, int, int, int] | None,
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
    native_output_hw = (
        decoder.cfg.output_grid[0] * decoder.cfg.output_patch_size,
        decoder.cfg.output_grid[1] * decoder.cfg.output_patch_size,
    )
    native_log_prediction = decoder(
        tokens,
        output_hw=native_output_hw,
        deterministic=True,
    )[..., 0]
    log_prediction = resize_log_depth_in_metric_space(
        native_log_prediction,
        output_hw=output_hw,
    )
    valid = depth_valid_mask(
        target_depth,
        min_depth=min_depth,
        max_depth=max_depth,
        eval_crop=eval_crop,
        valid_batch_size=valid_batch_size,
    )
    return depth_sufficient_statistics(
        log_prediction,
        target_depth,
        valid,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def run_validation(
    restored,
    decoder: PooledTaskDecoder,
    val_iter,
    *,
    mesh,
    prefetch: int,
    cfg: PooledDepthConfig,
) -> dict[str, float]:
    decoder.eval()
    totals: dict[str, float] = {}
    static = {
        **source_encoding_kwargs(restored),
        "output_hw": (cfg.data.height, cfg.data.width),
        "min_depth": cfg.data.min_depth,
        "max_depth": cfg.data.max_depth,
        "eval_crop": cfg.data.eval_crop,
    }
    for batch in prefetch_to_mesh(
        val_iter,
        prefetch,
        mesh,
        pad_to=mesh.size,
    ):
        statistics = depth_eval_step(
            restored.dino,
            restored.tokenizer,
            decoder,
            batch["image"],
            batch["depth"],
            batch["_valid_size"],
            **static,
        )
        for name, value in statistics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    summary = summarize_depth_statistics(
        totals,
        silog_lambda=cfg.data.silog_lambda,
    )
    summary["loss"] = (
        summary["silog"]
        + cfg.data.gradient_weight * summary["gradient_error"]
    )
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


def main(args: Args) -> dict[str, float]:
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
        output_channels=1,
        output_patch_size=restored.cfg.patch_size,
        head_bias=math.log(2.5),
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
    data_cfg = DepthDataConfig(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        silog_lambda=args.silog_lambda,
        gradient_weight=args.gradient_weight,
        eval_crop=(45, 471, 41, 601) if args.use_eval_crop else None,
        normalization_mean=tuple(restored.data_cfg.normalization_mean),
        normalization_std=tuple(restored.data_cfg.normalization_std),
        color_jitter=args.color_jitter,
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
    data = create_depth_dataloaders(
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

    default_path = f"output/depth/{args.experiment}"
    restore_manager, restore_step = open_restore_manager(
        args.restore,
        args.maybe_restore,
        default_path=default_path,
        gcs_bucket=args.gcs_bucket,
        item_names=DEPTH_ITEM_NAMES,
        restore_best=args.restore_best,
        best_metric_key="abs_rel",
        best_mode="min",
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
    cfg = PooledDepthConfig(
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
        DEPTH_ITEM_NAMES,
        save_interval_steps=val_interval,
        total_steps=total_updates,
        best_fn=lambda metrics: metrics["abs_rel"],
        best_mode="min",
        # Keep both the latest/final state and the best validation state.
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
        "NYUv2 source=%s unique_grid=%s pool=%s decoder_params=%.2fM "
        "updates=%d grad_acc=%d eval_crop=%s vit_init=%s",
        args.source_checkpoint,
        restored.latent_spec.grid,
        restored.cfg.pool_window,
        parameter_count / 1e6,
        total_updates,
        grad_acc_steps,
        data_cfg.eval_crop,
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
        depth_train_step,
        restored.dino,
        restored.tokenizer,
        decoder,
        optimizer,
    )
    static = {
        **source_encoding_kwargs(restored),
        "output_hw": (data_cfg.height, data_cfg.width),
        "min_depth": data_cfg.min_depth,
        "max_depth": data_cfg.max_depth,
        "silog_lambda": data_cfg.silog_lambda,
        "gradient_weight": data_cfg.gradient_weight,
    }
    micro_step = updates_completed * grad_acc_steps
    final_summary: dict[str, float] | None = None
    best_abs_rel = (
        float(wandb.run.summary.get("best_val_abs_rel", float("inf")))
        if use_wandb and wandb.run is not None
        else float("inf")
    )
    start_time = time.perf_counter()

    for batch in prefetch_to_mesh(data_iter, args.prefetch, ctx.mesh):
        if updates_completed >= total_updates:
            break
        micro_step += 1
        metrics = train_step_cached(batch["image"], batch["depth"], **static)
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
                    "train/silog": float(metrics["silog"]),
                    "train/gradient_error": float(metrics["gradient_error"]),
                    "train/abs_rel": float(metrics["abs_rel"]),
                    "train/rmse": float(metrics["rmse"]),
                    "train/log_rmse": float(metrics["log_rmse"]),
                    "train/delta1": float(metrics["delta1"]),
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
            best_abs_rel = min(best_abs_rel, final_summary["abs_rel"])
            logging.info(
                "step=%d val_loss=%.5f AbsRel=%.5f RMSE=%.5f "
                "logRMSE=%.5f delta1=%.4f",
                updates_completed,
                final_summary["loss"],
                final_summary["abs_rel"],
                final_summary["rmse"],
                final_summary["log_rmse"],
                final_summary["delta1"],
            )
            if use_wandb and updates_completed > wandb_resume_step:
                wandb.log(
                    {
                        "val/loss": final_summary["loss"],
                        "val/silog": final_summary["silog"],
                        "val/gradient_error": final_summary["gradient_error"],
                        "val/abs_rel": final_summary["abs_rel"],
                        "val/best_abs_rel": best_abs_rel,
                        "val/rmse": final_summary["rmse"],
                        "val/log_rmse": final_summary["log_rmse"],
                        "val/delta1": final_summary["delta1"],
                        "val/delta2": final_summary["delta2"],
                        "val/delta3": final_summary["delta3"],
                        "val/valid_pixels": final_summary["valid_pixels"],
                        "step": updates_completed,
                    }
                )
                wandb.run.summary["best_val_abs_rel"] = best_abs_rel
            save_checkpoint(
                checkpoint_manager,
                updates_completed,
                decoder,
                optimizer,
                data_iter,
                cfg,
                metrics={
                    "abs_rel": float(final_summary["abs_rel"]),
                    "rmse": float(final_summary["rmse"]),
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
