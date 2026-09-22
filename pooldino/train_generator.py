"""Flow matching on pooled-decoder latents.

This trains the class-conditional RAEv2 DDT on the compressed token field
produced by a trained ``train_pooled_decoder`` tokenizer. For repeat-pool
decoders the generator targets the pre-repeat pooled grid; generated latents
are repeated only at image-decoding time.

Example:

    python -m pooldino.train_generator \
        --pooled-decoder-path output/decoders/repeatconv4x4-dinol-vitxl-raev2official-tfds \
        --experiment raev2ddt-autokappa
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import chex
import flax.nnx as nnx
import grain.python as grain
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import jmp
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as pp
import tyro
import wandb
from absl import logging
from dacite import Config as DaciteConfig, from_dict
from tqdm import tqdm

from pooldino.data import (
    DataConfig,
    DataLoaders,
    RAEv2EpochSemantics,
    create_dataloaders,
    raev2_steps_per_epoch,
    raev2_total_updates,
)
from pooldino.experiment import ExperimentSpec, flag, option, positional
from pooldino.models.transformer import set_attn_implementation
from pooldino.pooled_generator import (
    RAEV2_CFG_DROPOUT_PROB,
    RAEV2_SELF_REPA_COEFF,
    RAEV2_SELF_REPA_LAYER,
    RAEV2_TIME_DIST_SHIFT_DIM,
    PooledGeneratorConfig,
    PooledLatentStatsIdentity,
    RestoredPooledDecoderComponents,
    bind_pooled_generator_artifacts,
    configure_for_pooled_latents,
    encode_pooled_decoder_targets_impl,
    load_pooled_latent_stats,
    make_pooled_generator,
    restore_pooled_decoder_components,
    restore_pooled_generator_item,
    validate_pooled_generator_artifacts,
)
from pooldino.models.raev2_ddt import RAEv2DDT
from pooldino.gmuon import (
    gmuon_with_adamw_fallback,
    optimizer_partition_counts,
    raev2_ddt_optimizer_labels,
)
from pooldino.augmentations.decoder import RAEv2GithubDecoderAugmentations
from pooldino.training import build_lr_schedule, init_wandb
from pooldino.utils import (
    TrainingProfiler,
    determine_save_path,
    init_distributed,
    is_primary_host,
    open_restore_manager,
    prefetch_to_mesh,
    restore_data_loader,
    restore_optimizer_state,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

RAEV2_CHECKPOINT_EPOCHS = (20, 80)


@dataclass
class Args:
    pooled_decoder_path: Path
    seed: int = 42
    epochs: int | None = None
    restore: Path | Literal["default"] | None = None
    maybe_restore: Path | Literal["default"] | None = None
    checkpoint: bool = True
    checkpoint_dir: Path | None = None
    success_marker: Path | None = None
    gcs_bucket: str | None = None
    data_in_bucket: bool = False
    num_data_workers: int | None = None
    use_wandb: bool = False
    wandb_name: str | None = None
    wandb_run_id: str | None = None
    wandb_log_every: int = 20
    project_name: str = "pooldino-pooled-generator"
    gpu_batch_size: int = 128
    save_epochs_freq: int = 5
    experiment: str = "raev2ddt"
    dataset: str | None = None
    implementation: Literal["cudnn", "xla"] = "cudnn"
    fsdp: int = 1
    distributed: bool = False
    checkpoints_to_keep: tuple[int, ...] | None = None
    stats_path: Path | None = None
    pooled_decoder_step: int | None = None
    use_ema_pooled_decoder: bool = True
    profile_mode: Literal["disabled", "always", "window"] = "disabled"
    profiler_port: int = 7777
    profiler_start_step: int = 10
    profiler_stop_step: int = 2000


def _cosine_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
    y = y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), 1e-6)
    return 1.0 - jnp.sum(x * y, axis=-1)


def _sample_time(
    key: jax.Array,
    batch_size: int,
    dtype,
    *,
    mu: float,
    sigma: float,
    kappa: float | None,
) -> jax.Array:
    logits = mu + sigma * jax.random.normal(key, (batch_size,), dtype=dtype)
    return _shift_sampled_time(jax.nn.sigmoid(logits), kappa)


def _shift_sampled_time(
    t: jax.Array,
    kappa: float | None,
) -> jax.Array:
    if kappa is None:
        return t
    # Released RAEv2 Transport.sample shifts official data->noise time.
    return kappa * t / (1.0 + (kappa - 1.0) * t)


def _raev2_training_path(
    data: jax.Array,
    noise: jax.Array,
    official_t: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Construct the released RAEv2 path, target drift, and safe time."""

    t = official_t.reshape(
        (official_t.shape[0],) + (1,) * (data.ndim - 1)
    )
    safe_t = jnp.maximum(t, jnp.asarray(eps, dtype=t.dtype))
    xt = (1.0 - t) * data + t * noise
    # This deliberately is not simply ``noise - data`` below t_eps: the
    # released transport divides both target and x-pred conversion by the
    # clamped time.
    target_drift = (xt - data) / safe_t
    return xt, target_drift, safe_t


def _apply_transport_class_dropout(
    labels: jax.Array,
    key: jax.Array,
    probability: float,
    num_classes: int,
) -> tuple[jax.Array, jax.Array]:
    """Apply the released transport-level classifier-free dropout."""
    drop = jax.random.bernoulli(key, probability, shape=labels.shape)
    dropped = jnp.where(drop, num_classes, labels)
    return dropped, jnp.mean(drop.astype(jnp.float32))


@nnx.jit(
    static_argnames=(
        "should_update_ema",
        "ema_momentum",
        "kappa",
        "xpred_denom_eps",
        "conditioning_dropout_prob",
        "num_classes",
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
        "time_mu",
        "time_sigma",
        "base_model_coeff",
        "use_self_repa",
        "self_repa_coeff",
        "self_repa_cosine_weight",
    ),
    donate_argnames=("generator", "generator_ema", "optim"),
)
def train_step(
    generator: RAEv2DDT,
    generator_ema: RAEv2DDT,
    dino,
    tokenizer,
    optim: nnx.Optimizer,
    images: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    ema_momentum: float,
    *,
    should_update_ema: bool,
    kappa: float | None,
    xpred_denom_eps: float,
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
    time_mu: float,
    time_sigma: float,
    base_model_coeff: float,
    use_self_repa: bool,
    self_repa_coeff: float,
    self_repa_cosine_weight: float,
    conditioning_dropout_prob: float = RAEV2_CFG_DROPOUT_PROB,
    num_classes: int = 1000,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    chex.assert_rank(images, 4)
    latents, self_repa_target = encode_pooled_decoder_targets_impl(
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
    latents = generator.normalize(latents)

    key_noise, key_time, key_dropout = jax.random.split(key, 3)
    noise = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    sampled_t = _sample_time(
        key_time,
        latents.shape[0],
        latents.dtype,
        mu=time_mu,
        sigma=time_sigma,
        kappa=kappa,
    )
    model_t = sampled_t
    interpolation_t = 1.0 - model_t
    xt, target_velocity, transport_denom = _raev2_training_path(
        latents,
        noise,
        model_t,
        xpred_denom_eps,
    )

    model_labels, label_drop_fraction = _apply_transport_class_dropout(
        labels,
        key_dropout,
        conditioning_dropout_prob,
        num_classes,
    )

    def loss_fn(model: RAEv2DDT):
        prediction = model(
            xt,
            model_t,
            model_labels,
            train=True,
            return_base_model=True,
            return_self_repa=use_self_repa,
        )
        predicted_velocity = (xt - prediction["x"]) / transport_denom
        flow_loss = jnp.mean(jnp.square(predicted_velocity - target_velocity))

        base_velocity = (xt - prediction["base_x"]) / transport_denom
        base_model_loss = jnp.mean(jnp.square(base_velocity - target_velocity))

        self_repa_loss = jnp.asarray(0.0, dtype=flow_loss.dtype)
        self_repa_cosine = jnp.asarray(0.0, dtype=flow_loss.dtype)
        if use_self_repa:
            self_repa_prediction = prediction["self_repa"]
            if self_repa_prediction.shape != self_repa_target.shape:
                raise ValueError(
                    f"Self-REPA prediction {self_repa_prediction.shape} does not match "
                    f"the frozen target {self_repa_target.shape}."
                )
            self_repa_loss = jnp.mean(
                jnp.square(self_repa_prediction - self_repa_target)
            )
            self_repa_cosine = jnp.mean(
                _cosine_distance(self_repa_prediction, self_repa_target)
            )

        total = flow_loss + base_model_coeff * base_model_loss
        total = total + self_repa_coeff * (
            self_repa_loss
            + self_repa_cosine_weight * self_repa_cosine
        )
        return total, {
            "flow_loss": flow_loss,
            "base_model_loss": base_model_loss,
            "self_repa_loss": self_repa_loss,
            "self_repa_cosine": self_repa_cosine,
            "t_mean": jnp.mean(interpolation_t),
            "model_t_mean": jnp.mean(model_t),
            "label_drop_fraction": label_drop_fraction,
            "latent_mean": jnp.mean(latents),
            "latent_std": jnp.std(latents),
        }

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(generator)
    optim.update(generator, grads)

    if should_update_ema:
        new_ema_state = jax.tree.map(
            lambda target, source: target * ema_momentum + source * (1.0 - ema_momentum),
            nnx.state(generator_ema, nnx.Param),
            nnx.state(generator, nnx.Param),
        )
        nnx.update(generator_ema, new_ema_state)
    return loss, metrics


def save_checkpoint(
    manager: ocp.CheckpointManager | None,
    step: int,
    generator: RAEv2DDT,
    generator_ema: RAEv2DDT,
    optim: nnx.Optimizer,
    data_iter,
    cfg: PooledGeneratorConfig,
) -> bool:
    if manager is None:
        return False
    return manager.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(generator))),
            model_ema=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(generator_ema))),
            optim=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim))),
            loader=grain.PyGrainCheckpointSave(data_iter),
            config=ocp.args.JsonSave(asdict(cfg)),
        ),
    )


class RAEv2DDTExperiment(
    ExperimentSpec,
    examples=(
        "raev2ddt",
        "raev2ddt-autokappa",
        "raev2ddt-kappa8.0",
        "raev2ddt-nosrepa",
        "raev2ddt-srepa1.0-srl8",
    ),
):
    """RAEv2 reference recipe plus self-REPA pooling ablations."""

    architecture: Literal["ddt"] = positional(prefix="raev2")
    autokappa: bool = flag()
    nosrepa: bool = flag()
    kappa: float | None = option()
    srepa: float | None = option()
    srl: int | None = option()
    srepacos: float | None = option()


def _get_raev2_ddt_experiment(
    name: str,
    restored: RestoredPooledDecoderComponents,
) -> PooledGeneratorConfig:
    stage1_profile = getattr(
        getattr(restored, "cfg", None),
        "stage1_profile",
        None,
    )
    if stage1_profile != "raev2_github":
        raise ValueError(
            "Exact raev2ddt experiments require a pooled decoder trained with "
            "stage1_profile='raev2_github'; "
            f"got {stage1_profile!r}."
        )
    spec = RAEv2DDTExperiment.parse_or_raise(name)
    if spec.kappa is not None and spec.autokappa:
        raise ValueError("-kappa and -autokappa are mutually exclusive.")
    if spec.srepa is not None and spec.srepa <= 0:
        raise ValueError("-srepa coefficient must be positive.")
    if spec.srepacos is not None and spec.srepacos < 0:
        raise ValueError("-srepacos must be non-negative.")
    if spec.nosrepa and any(
        value is not None for value in (spec.srepa, spec.srl, spec.srepacos)
    ):
        raise ValueError("-nosrepa cannot be combined with self-REPA overrides.")

    baseline = PooledGeneratorConfig()
    self_repa_layer = spec.srl or RAEV2_SELF_REPA_LAYER
    if self_repa_layer > baseline.raev2_ddt.encoder_depth:
        raise ValueError(
            f"-srl{self_repa_layer} exceeds the "
            f"{baseline.raev2_ddt.encoder_depth}-block encoder."
        )

    cfg = replace(
        baseline,
        raev2_ddt=replace(
            baseline.raev2_ddt,
            input_grid=restored.latent_spec.grid,
            in_channels=restored.latent_spec.feat,
        ),
        time_dist_shift_dim=(
            None
            if spec.autokappa or spec.kappa is not None
            else RAEV2_TIME_DIST_SHIFT_DIM
        ),
        kappa=spec.kappa,
        self_repa=not spec.nosrepa,
        self_repa_layer=self_repa_layer,
        self_repa_coeff=(
            spec.srepa if spec.srepa is not None else RAEV2_SELF_REPA_COEFF
        ),
        self_repa_cosine_weight=spec.srepacos or 0.0,
        self_repa_target_grid=restored.grid_hw,
    )
    return configure_for_pooled_latents(cfg, restored.latent_spec)


def get_experiment(
    name: str,
    restored: RestoredPooledDecoderComponents,
) -> PooledGeneratorConfig:
    return _get_raev2_ddt_experiment(name, restored)


def _generator_config_mismatch_paths(
    expected: object,
    actual: object,
    prefix: str = "",
) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                mismatches.append(path)
            else:
                mismatches.extend(
                    _generator_config_mismatch_paths(expected[key], actual[key], path)
                )
        return mismatches
    return [] if expected == actual else [prefix]


def resolve_pooled_generator_restore_config(
    target_cfg: PooledGeneratorConfig,
    restored_cfg: PooledGeneratorConfig,
) -> PooledGeneratorConfig:
    """Reject an exact checkpoint restored under a different experiment name.

    Artifact identities and optimizer partitions are bound at runtime and are
    validated independently after stats/model restoration. Every experiment
    semantic remains part of this pre-restore contract.
    """
    if target_cfg.train.epochs < restored_cfg.train.epochs:
        raise ValueError(
            "Cannot resume a pooled generator with a shorter training horizon: "
            f"requested={target_cfg.train.epochs}, "
            f"checkpoint={restored_cfg.train.epochs}."
        )
    comparison_cfg = restored_cfg
    if target_cfg.train.epochs != restored_cfg.train.epochs:
        comparison_cfg = replace(
            restored_cfg,
            train=replace(restored_cfg.train, epochs=target_cfg.train.epochs),
        )

    def semantic_value(cfg: PooledGeneratorConfig) -> dict[str, object]:
        return asdict(
            replace(
                cfg,
                pooled_decoder_identity=None,
                latent_stats_identity=None,
                optimizer_partition_counts=None,
            )
        )

    mismatches = _generator_config_mismatch_paths(
        semantic_value(target_cfg),
        semantic_value(comparison_cfg),
    )
    if mismatches:
        preview = ", ".join(mismatches[:16])
        if len(mismatches) > 16:
            preview += f", ... ({len(mismatches)} total)"
        raise ValueError(
            "Exact RAEv2 stage-two restore does not match the requested "
            f"experiment contract: {preview}. Use the checkpoint's experiment "
            "name or create an explicit fork/output."
        )
    return comparison_cfg


def _load_latent_stats(
    stats_path: Path,
    eps: float,
    *,
    restored: RestoredPooledDecoderComponents,
) -> tuple[
    jax.Array,
    jax.Array,
    PooledLatentStatsIdentity,
]:
    try:
        mean_np, var_np, identity = load_pooled_latent_stats(
            stats_path,
            expected_shape=(
                restored.latent_spec.num_latents,
                restored.latent_spec.feat,
            ),
            expected_decoder_identity=restored.identity,
            require_source_metadata=True,
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Latent normalization is enabled but stats are missing: {stats_path}. "
            "Run pooldino.compute_stats first."
        ) from error
    mean = jnp.asarray(mean_np, dtype=jnp.float32)
    var = jnp.asarray(var_np, dtype=jnp.float32)
    std = jnp.sqrt(var + eps)
    return mean, std, identity


def build_optimizer_transform(
    cfg: PooledGeneratorConfig,
    learning_rate,
) -> optax.GradientTransformation:
    """Build the persisted optimizer recipe, before accumulation wrapping."""

    base = gmuon_with_adamw_fallback(
        learning_rate,
        adam_b1=cfg.train.adam_b1,
        adam_b2=cfg.train.adam_b2,
        adam_eps=1e-8,
        momentum=cfg.train.momentum,
        nesterov=cfg.train.nesterov,
        weight_decay=cfg.train.weight_decay,
        label_fn=raev2_ddt_optimizer_labels,
    )
    return optax.chain(optax.clip_by_global_norm(1.0), base)


def resolve_optimizer_partition_counts(
    cfg: PooledGeneratorConfig,
    params,
) -> dict[str, int]:
    """Summarize the actual persisted optimizer split for this model tree."""

    return optimizer_partition_counts(params, raev2_ddt_optimizer_labels)


def _resolve_generator_data_config(
    restored: RestoredPooledDecoderComponents,
    *,
    dataset: str | None,
    num_workers: int | None,
) -> DataConfig:
    """Resolve stage-two data without losing the released stage-one backend."""

    data_cfg = restored.data_cfg
    if dataset is not None:
        data_cfg = DataConfig.from_preset(
            dataset,
            normalization_mean=data_cfg.normalization_mean,
            normalization_std=data_cfg.normalization_std,
        )
    if num_workers is not None:
        data_cfg = replace(data_cfg, num_workers=num_workers)
    return data_cfg


def resolve_training_steps(
    train_ds_size: int,
    cfg: PooledGeneratorConfig,
    *,
    grad_acc_steps: int = 1,
) -> tuple[int, int]:
    """Return total updates and epoch width under the selected recipe."""

    if train_ds_size <= 0 or cfg.train.batch_size <= 0:
        raise ValueError("Dataset and global batch sizes must be positive.")
    steps_per_epoch = raev2_steps_per_epoch(
        train_ds_size,
        cfg.train.batch_size,
        world_size=8,
        grad_accum_steps=grad_acc_steps,
    )
    total_updates = raev2_total_updates(
        train_ds_size,
        cfg.train.batch_size,
        cfg.train.epochs,
        world_size=8,
        grad_accum_steps=grad_acc_steps,
    )
    return total_updates, steps_per_epoch


def resolve_train_epoch_semantics(
    cfg: PooledGeneratorConfig,
    *,
    grad_acc_steps: int,
) -> RAEv2EpochSemantics:
    """Match the released eight-rank epoch-tail truncation semantics."""

    return RAEv2EpochSemantics(
        global_batch_size=cfg.train.batch_size,
        source_world_size=8,
        grad_accum_steps=grad_acc_steps,
    )


def resolve_checkpoints_to_keep(
    cfg: PooledGeneratorConfig,
    requested_epochs: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve protected epoch checkpoints for the selected recipe."""

    if requested_epochs is None:
        return RAEV2_CHECKPOINT_EPOCHS

    epochs = tuple(dict.fromkeys(requested_epochs))
    invalid = [epoch for epoch in epochs if not 1 <= epoch <= cfg.train.epochs]
    if invalid:
        raise ValueError(
            "Checkpoint epochs must lie within the training run; "
            f"got {invalid} for {cfg.train.epochs} epochs."
        )
    missing = sorted(set(RAEV2_CHECKPOINT_EPOCHS) - set(epochs))
    if missing:
        raise ValueError(
            "RAEv2 runs must preserve checkpoints at epochs 20 and 80; "
            f"missing {missing}."
        )
    return epochs


def write_success_marker(
    save_path: Path | str | None,
    marker_path: Path | None = None,
) -> None:
    """Create a local marker used by retry wrappers to detect completed runs."""

    if marker_path is None:
        if save_path is None:
            return
        if isinstance(save_path, str):
            if save_path.startswith("gs://"):
                logging.warning("Skipping local _SUCCESS marker for GCS checkpoint path.")
                return
            marker_dir = Path(save_path)
        else:
            marker_dir = save_path
        marker_path = marker_dir / "_SUCCESS"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.touch()


def main(args: Args) -> None:
    if args.distributed:
        init_distributed()
    if not is_primary_host():
        logging.set_verbosity(logging.WARNING)

    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )
    num_devices = jax.device_count()
    if num_devices % args.fsdp != 0:
        raise ValueError(f"Devices ({num_devices}) must be divisible by fsdp ({args.fsdp}).")
    data_parallel_size = num_devices // args.fsdp
    devices = mesh_utils.create_device_mesh((data_parallel_size, args.fsdp))
    mesh = Mesh(devices, ("data", "model"))
    jax.set_mesh(mesh)
    rngs = nnx.Rngs(args.seed + jax.process_index())

    pooled_decoder_path: Path | str = (
        f"gs://{args.gcs_bucket}/{args.pooled_decoder_path}"
        if args.gcs_bucket is not None
        else args.pooled_decoder_path
    )
    restored = restore_pooled_decoder_components(
        pooled_decoder_path,
        mesh=mesh,
        mp=mp,
        seed=args.seed,
        step=args.pooled_decoder_step,
        use_ema=args.use_ema_pooled_decoder,
        restore_decoder=False,
        implementation=args.implementation,
        source_encoder_fp32=True,
    )
    target_cfg = get_experiment(args.experiment, restored)
    if args.epochs is not None:
        target_cfg = replace(
            target_cfg,
            train=replace(target_cfg.train, epochs=args.epochs),
        )
    cfg = target_cfg

    pooled_name = Path(str(args.pooled_decoder_path)).name
    default_output = f"output/generators/{pooled_name}/{args.experiment}"
    item_names = ["model", "model_ema", "optim", "loader", "config"]
    restore_mngr, restore_step = open_restore_manager(
        args.restore,
        args.maybe_restore,
        default_output,
        args.gcs_bucket,
        item_names,
    )
    if restore_mngr is not None and restore_step > 0:
        cfg_raw = restore_mngr.restore(
            restore_step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]
        restored_cfg = from_dict(
            PooledGeneratorConfig,
            cfg_raw,
            config=DaciteConfig(cast=[tuple], strict=False),
        )
        restored_cfg = configure_for_pooled_latents(
            restored_cfg,
            restored.latent_spec,
        )
        cfg = resolve_pooled_generator_restore_config(target_cfg, restored_cfg)

    if cfg.train.batch_size % (args.gpu_batch_size * data_parallel_size) != 0:
        raise ValueError("Global batch size must be divisible by the aggregate micro batch.")
    grad_acc_steps = cfg.train.batch_size // (args.gpu_batch_size * data_parallel_size)
    micro_bs = args.gpu_batch_size * data_parallel_size
    logging.info("Gradient accumulation steps: %d", grad_acc_steps)
    logging.info("Micro batch size: %d", micro_bs)
    logging.info(
        "Pooled latents: grid=%s tokens=%d feat=%d",
        restored.latent_spec.grid,
        restored.latent_spec.num_latents,
        restored.latent_spec.feat,
    )

    data_cfg = _resolve_generator_data_config(
        restored,
        dataset=args.dataset,
        num_workers=args.num_data_workers,
    )
    train_aug = RAEv2GithubDecoderAugmentations(restored.aug_cfg, data_cfg)
    val_aug = RAEv2GithubDecoderAugmentations(restored.aug_cfg, data_cfg)
    data: DataLoaders = create_dataloaders(
        data_cfg,
        micro_bs,
        train_aug=train_aug,
        val_aug=val_aug,
        val_epochs=1,
        drop_remainder_train=True,
        drop_remainder_val=True,
        gcs_bucket=args.gcs_bucket if args.data_in_bucket else None,
        train_epoch_semantics=resolve_train_epoch_semantics(
            cfg,
            grad_acc_steps=grad_acc_steps,
        ),
    )
    total_updates, steps_per_epoch = resolve_training_steps(
        data.train_ds_size,
        cfg,
        grad_acc_steps=grad_acc_steps,
    )
    data_iter = iter(data.train_loader)
    lr_sched = build_lr_schedule(cfg.train, total_updates)

    save_path = determine_save_path(
        checkpoint_enabled=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        default_path=default_output,
        gcs_bucket=args.gcs_bucket,
    )
    stats_path = args.stats_path or (
        Path(str(args.pooled_decoder_path)) / "pooled_latent_stats.npz"
    )
    latent_mean, latent_std, stats_identity = _load_latent_stats(
        stats_path,
        cfg.latent_norm_eps,
        restored=restored,
    )

    resuming = restore_mngr is not None and restore_step > 0
    if resuming:
        validate_pooled_generator_artifacts(cfg, restored, stats_identity)
    else:
        cfg = bind_pooled_generator_artifacts(cfg, restored, stats_identity)

    if resuming:
        def restore_model(item: str):
            return restore_pooled_generator_item(
                restore_mngr,
                restore_step,
                item,
                cfg=cfg,
                latent_spec=restored.latent_spec,
                mesh=mesh,
                mp=mp,
                latent_mean=latent_mean,
                latent_std=latent_std,
            )

        generator = restore_model("model")
        generator_ema = restore_model("model_ema")
    else:
        generator = make_pooled_generator(
            cfg,
            restored.latent_spec,
            mp,
            rngs,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        generator_ema = make_pooled_generator(
            cfg,
            restored.latent_spec,
            mp,
            rngs,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        nnx.update(
            generator_ema,
            jax.tree.map(lambda value: jnp.copy(value), nnx.state(generator)),
        )

    set_attn_implementation(generator, args.implementation)
    set_attn_implementation(generator_ema, args.implementation)
    partition_counts = resolve_optimizer_partition_counts(
        cfg,
        nnx.state(generator, nnx.Param),
    )
    if (
        cfg.optimizer_partition_counts is not None
        and cfg.optimizer_partition_counts != partition_counts
    ):
        raise ValueError(
            "Optimizer partition does not match the generator checkpoint: "
            f"checkpoint={cfg.optimizer_partition_counts}, actual={partition_counts}."
        )
    cfg = replace(cfg, optimizer_partition_counts=partition_counts)
    logging.info("Optimizer partition: %s", partition_counts)
    decoder_run_id = Path(str(args.pooled_decoder_path)).name
    use_wandb, wandb_resume_step = init_wandb(
        args,
        cfg,
        project_name=args.project_name,
        run_id_suffix=f"{decoder_run_id}-s{args.seed}",
    )

    optimizer_chain = build_optimizer_transform(cfg, lr_sched)
    optimizer_chain = optax.MultiSteps(optimizer_chain, grad_acc_steps)
    optim = nnx.Optimizer(generator, optimizer_chain, wrt=nnx.Param)
    if restore_mngr is not None:
        restore_optimizer_state(restore_mngr, restore_step, optim, mesh)
        data_iter = restore_data_loader(restore_mngr, restore_step, data_iter)

    parameter_count = sum(p.size for p in jax.tree.leaves(nnx.state(generator, nnx.Param)))
    logging.info("Generator parameters: %.2fM", parameter_count / 1_000_000)

    if save_path is not None:
        path_str = str(save_path.absolute()) if isinstance(save_path, Path) else save_path
        save_interval_steps = steps_per_epoch * max(1, args.save_epochs_freq)
        save_on_steps = frozenset([total_updates])
        policies: list[pp.PreservationPolicy] = [pp.LatestN(1)]
        checkpoint_epochs = resolve_checkpoints_to_keep(
            cfg,
            args.checkpoints_to_keep,
        )
        if checkpoint_epochs:
            keep_steps = {epoch * steps_per_epoch for epoch in checkpoint_epochs}
            save_on_steps |= frozenset(keep_steps)
            policies.append(pp.CustomSteps(keep_steps))
        manager = ocp.CheckpointManager(
            path_str,
            item_names=item_names,
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=save_interval_steps,
                save_on_steps=save_on_steps,
                create=True,
                read_only=False,
                preservation_policy=pp.AnyPreservationPolicy(policies),
            ),
        )
    else:
        manager = None

    kappa = cfg.kappa
    if kappa is None and cfg.time_dist_shift_base is not None:
        shift_dim = cfg.time_dist_shift_dim
        if shift_dim is None:
            shift_dim = restored.latent_spec.num_latents * restored.latent_spec.feat
        kappa = max(1.0, (shift_dim / cfg.time_dist_shift_base) ** 0.5)
    logging.info(
        "Training: RAEv2 DDT x-prediction, logit-normal data-to-noise time; "
        "kappa=%s optimizer=%s wd=%s base=(depth=%s, coeff=%s) transport_eps=%s "
        "self_repa=(enabled=%s, layer=%s, coeff=%s, cosine=%s, target_grid=%s)",
        kappa,
        cfg.train.optimizer,
        cfg.train.weight_decay,
        cfg.base_model_depth,
        cfg.base_model_coeff,
        cfg.xpred_denom_eps,
        cfg.self_repa,
        cfg.self_repa_layer,
        cfg.self_repa_coeff,
        cfg.self_repa_cosine_weight,
        cfg.self_repa_target_grid,
    )

    updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
    pbar = tqdm(
        desc="Update",
        initial=updates_completed,
        total=total_updates,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        disable=not is_primary_host(),
    )
    profiler = TrainingProfiler(
        mode=args.profile_mode,
        port=args.profiler_port,
        start_step=args.profiler_start_step,
        stop_step=args.profiler_stop_step,
    )
    train_step_cached = nnx.cached_partial(
        train_step,
        generator,
        generator_ema,
        restored.dino,
        restored.tokenizer,
        optim,
    )
    base_key = rngs()
    micro_step = updates_completed * grad_acc_steps

    for samples in prefetch_to_mesh(data_iter, 1, mesh):
        micro_step += 1
        mini_step = int(optax.tree_utils.tree_get(optim.opt_state, "mini_step"))
        should_update_ema = (mini_step + 1) % grad_acc_steps == 0
        previous_updates = updates_completed

        train_loss, train_metrics = train_step_cached(
            samples["image"],
            samples["label"],
            jax.random.fold_in(base_key, micro_step),
            cfg.train.ema,
            should_update_ema=should_update_ema,
            kappa=kappa,
            xpred_denom_eps=cfg.xpred_denom_eps,
            conditioning_dropout_prob=cfg.conditioning_dropout_prob,
            num_classes=cfg.raev2_ddt.num_classes,
            backbone_resolution=restored.cfg.backbone_resolution,
            num_prefix_tokens=restored.cfg.num_prefix_tokens,
            layer_indices=restored.layer_indices,
            aggregation=restored.cfg.representation.aggregation,
            normalize_each=restored.cfg.representation.normalize_each,
            add_final_mean=restored.cfg.representation.add_final_mean,
            post_pool_norm=restored.cfg.post_pool_norm,
            representation_eps=restored.cfg.representation.eps,
            grid_hw=restored.grid_hw,
            pool_hw=restored.cfg.pool_window,
            time_mu=cfg.time_mu,
            time_sigma=cfg.time_sigma,
            base_model_coeff=cfg.base_model_coeff,
            use_self_repa=cfg.self_repa,
            self_repa_coeff=cfg.self_repa_coeff,
            self_repa_cosine_weight=cfg.self_repa_cosine_weight,
        )

        updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
        ran_update = updates_completed > previous_updates
        if ran_update:
            pbar.update(updates_completed - previous_updates)
            profiler.step(updates_completed)
            if manager and manager.reached_preemption(updates_completed):
                jax.block_until_ready(train_loss)
                save_checkpoint(
                    manager,
                    updates_completed,
                    generator,
                    generator_ema,
                    optim,
                    data_iter,
                    cfg,
                )
                manager.wait_until_finished()
                break
            if manager and manager.should_save(updates_completed):
                jax.block_until_ready(train_loss)
                save_checkpoint(
                    manager,
                    updates_completed,
                    generator,
                    generator_ema,
                    optim,
                    data_iter,
                    cfg,
                )

        if use_wandb and updates_completed > wandb_resume_step:
            if ran_update and updates_completed % args.wandb_log_every == 0:
                metrics = {
                    "train/loss": float(train_loss),
                    "train/lr": float(lr_sched(updates_completed)),
                    "step": updates_completed,
                }
                metrics.update(
                    {f"train/{name}": float(value) for name, value in train_metrics.items()}
                )
                wandb.log(metrics)

        if updates_completed >= total_updates:
            break

    completed = updates_completed >= total_updates
    pbar.close()
    if manager is not None:
        manager.close()
    if completed and is_primary_host():
        write_success_marker(save_path, args.success_marker)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(tyro.cli(Args))
