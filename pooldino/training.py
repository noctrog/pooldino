"""Shared training utilities for pooldino projects.

Provides helper functions that reduce boilerplate across training scripts
(decoder, generator, and downstream tasks) without imposing a framework or base class
for the training loop itself.
"""

import math
import hashlib
import sys
from dataclasses import dataclass, asdict, replace
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import grain.python as grain
import jmp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import wandb
from tqdm import tqdm
from dacite import from_dict, Config as DaciteConfig
from absl import logging

from pooldino.utils import (
    TrainingProfiler,
    git_commit_hash,
    is_primary_host,
    init_distributed,
    open_restore_manager,
)

T = TypeVar("T")
_BYTES_PER_GIB = 1024**3
_WANDB_TEXT_LIMIT = 128


# ---------------------------------------------------------------------------
# BaseArgs — shared CLI fields for all training scripts
# ---------------------------------------------------------------------------


@dataclass
class BaseArgs:
    seed: int = 42
    """Random seed for reproducibility."""
    restore: Path | Literal["default"] | None = None
    """Path to restore checkpoint from. Use 'default' to restore from the default
    checkpoint path (output/{project}/{experiment} or GCS equivalent if gcs_bucket is set)."""
    maybe_restore: Path | Literal["default"] | None = None
    """Like --restore, but if no checkpoint exists, logs a warning and trains from scratch."""
    restore_best: bool = False
    """Pick the highest-metric checkpoint instead of the latest.
    Requires --restore/--maybe-restore. Each project sets its own scoring metric."""
    checkpoint: bool = True
    """Enable checkpoint saving."""
    checkpoint_dir: Path | None = None
    """Override checkpoint directory (default: output/{project}/{experiment})."""
    keep_checkpoints_without_metrics: bool = True
    """Keep checkpoints even if no validation metrics were recorded at that step."""
    gcs_bucket: str | None = None
    """GCS bucket name for remote checkpointing (converts paths to gs://{bucket}/{path})."""
    data_in_bucket: bool = False
    """Load dataset from the GCS bucket instead of local disk.
    Requires --gcs-bucket. The bucket must mirror the local tfds directory layout:
    gs://{gcs_bucket}/{dataset_name}/{version}/ with dataset_info.json, features.json,
    and data shards. Prepare locally first, then upload with
    ``gsutil -m cp -r ~/tensorflow_datasets/{dataset} gs://{bucket}/{dataset}``."""
    num_data_workers: int | None = None
    """Number of grain data-loading workers. Overrides per-project default when set."""
    use_wandb: bool = False
    """Enable Weights & Biases logging."""
    wandb_run_id: str | None = None
    """Explicit W&B run ID, useful when resuming after a code commit."""
    wandb_log_every: int = 20
    """Log training metrics to wandb every N optimizer updates."""
    wandb_name: str | None = None
    """Custom wandb run name (defaults to --experiment)."""
    gpu_batch_size: int = 128
    """Per-device micro-batch size. Gradient accumulation fills the rest."""
    prefetch: int = 1
    """Number of batches to prefetch to device in background."""
    profile_mode: Literal["disabled", "always", "window"] = "disabled"
    """JAX profiler mode: 'disabled', 'always' (server runs entire training),
    or 'window' (active between profiler_start_step and profiler_stop_step)."""
    profiler_port: int = 7777
    """Port for the JAX profiler server."""
    profiler_start_step: int = 10
    """Step at which to start profiling (window mode only)."""
    profiler_stop_step: int = 2000
    """Step at which to stop profiling (window mode only)."""
    val_epochs_freq: int = 5
    """Run validation and save a checkpoint every N epochs."""
    experiment: str = "baseline"
    """Experiment name. Selects a static config or is parsed as a dynamic spec."""
    fsdp: int = 1
    """Number of devices for FSDP model sharding (rest used for data parallelism)."""
    distributed: bool = False
    """Enable multi-host distributed training (calls jax.distributed.initialize)."""


# ---------------------------------------------------------------------------
# TrainingContext — groups mesh/mp/rngs from setup_context
# ---------------------------------------------------------------------------


@dataclass
class TrainingContext:
    mesh: Mesh
    mp_policy: jmp.Policy
    rngs: nnx.Rngs
    data_parallel_size: int


def setup_context(args: BaseArgs) -> TrainingContext:
    """Distributed init, logging, MP policy, rngs, mesh creation, jax.set_mesh.

    Call this at the start of ``main()`` before building models or data loaders.
    """
    if args.distributed:
        init_distributed()
    if not is_primary_host():
        logging.set_verbosity(logging.WARNING)

    mp_policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32
    )
    rngs = nnx.Rngs(args.seed + jax.process_index())

    num_devices = jax.device_count()
    if num_devices % args.fsdp != 0:
        raise ValueError(
            f"Number of devices ({num_devices}) must be divisible by fsdp ({args.fsdp})"
        )
    data_parallel_size = num_devices // args.fsdp
    devices = mesh_utils.create_device_mesh((data_parallel_size, args.fsdp))
    mesh = Mesh(devices, ("data", "model"))
    jax.set_mesh(mesh)
    logging.info(f"Process count: {jax.process_count()}")
    logging.info(f"Mesh shape: data={data_parallel_size}, model={args.fsdp}")

    return TrainingContext(
        mesh=mesh,
        mp_policy=mp_policy,
        rngs=rngs,
        data_parallel_size=data_parallel_size,
    )


# ---------------------------------------------------------------------------
# Checkpoint restore + config merge
# ---------------------------------------------------------------------------


def restore_and_merge_config(
    args: BaseArgs,
    cfg,
    cfg_class: type[T],
    *,
    default_path: str,
    item_names: list[str],
    best_metric_key: str | None = None,
    best_mode: Literal["max", "min"] = "max",
) -> tuple[ocp.CheckpointManager | None, int, T]:
    """Open a restore manager, load checkpoint config, and merge training config.

    Pass ``best_metric_key`` to enable ``--restore-best`` for this project.
    Returns ``(restore_mngr, ckpt_step, merged_cfg)``.
    """
    restore_mngr, ckpt_step = open_restore_manager(
        args.restore,
        args.maybe_restore,
        default_path=default_path,
        gcs_bucket=args.gcs_bucket,
        item_names=item_names,
        restore_best=args.restore_best,
        best_metric_key=best_metric_key,
        best_mode=best_mode,
    )

    if restore_mngr is not None:
        cfg_d = restore_mngr.restore(
            ckpt_step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )["config"]
        ckpt_cfg = from_dict(cfg_class, cfg_d, DaciteConfig(cast=[tuple], strict=False))
        target_cfg = cfg
        if ckpt_cfg.train != target_cfg.train:
            logging.warning("Restoring model/data from checkpoint but overriding training config.")
        if ckpt_cfg.train.batch_size != target_cfg.train.batch_size:
            raise ValueError("Restoring with a different batch_size is not supported.")
        cfg = replace(ckpt_cfg, train=target_cfg.train)

    return restore_mngr, ckpt_step, cfg


# ---------------------------------------------------------------------------
# Wandb init with resume support
# ---------------------------------------------------------------------------


def _memory_metric_stem(stat_name: str) -> str:
    stem = stat_name
    if stem.startswith("bytes_"):
        stem = stem.removeprefix("bytes_")
    stem = stem.replace("_bytes_", "_")
    if stem.endswith("_bytes"):
        stem = stem.removesuffix("_bytes")
    return stem


def _short_wandb_text(text: str, *, limit: int = _WANDB_TEXT_LIMIT) -> str:
    """Shorten wandb name/id fields while keeping a stable uniqueness suffix."""
    if len(text) <= limit:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    suffix = f"-{digest}"
    return f"{text[: max(1, limit - len(suffix))]}{suffix}"


def jax_system_metrics(
    *,
    prefix: str = "system_metrics",
    devices: Iterable[Any] | None = None,
) -> dict[str, float]:
    """Return JAX allocator memory metrics for wandb logging.

    Device memory stats are backend-dependent. GPU/TPU backends may return
    allocator counters, while CPU commonly returns ``None``.
    """
    local_devices = list(jax.local_devices() if devices is None else devices)
    metrics: dict[str, float] = {}
    byte_stats: dict[str, list[float]] = {}
    dynamic_byte_stats = {
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_reserved",
        "peak_bytes_reserved",
    }
    for device in local_devices:
        memory_stats = getattr(device, "memory_stats", None)
        if memory_stats is None:
            continue
        try:
            stats = memory_stats()
        except Exception:
            continue
        if not stats:
            continue
        for name, value in stats.items():
            if name not in dynamic_byte_stats and name != "bytes_limit":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            byte_stats.setdefault(name, []).append(numeric)

    for name, values in byte_stats.items():
        if name == "bytes_limit" or not values:
            continue
        stem = _memory_metric_stem(name)
        metrics[f"{prefix}/jax_memory_{stem}_max_gib"] = max(values) / _BYTES_PER_GIB
        metrics[f"{prefix}/jax_memory_{stem}_total_gib"] = sum(values) / _BYTES_PER_GIB

    in_use = byte_stats.get("bytes_in_use")
    limit = byte_stats.get("bytes_limit")
    if in_use and limit:
        pairs = [(used, cap) for used, cap in zip(in_use, limit, strict=False) if cap > 0]
        if pairs:
            metrics[f"{prefix}/jax_memory_utilization_max"] = max(
                used / cap for used, cap in pairs
            )
            total_limit = sum(cap for _, cap in pairs)
            if total_limit > 0:
                metrics[f"{prefix}/jax_memory_utilization_total"] = (
                    sum(used for used, _ in pairs) / total_limit
                )

    peak_in_use = byte_stats.get("peak_bytes_in_use")
    if peak_in_use and limit:
        pairs = [(used, cap) for used, cap in zip(peak_in_use, limit, strict=False) if cap > 0]
        if pairs:
            metrics[f"{prefix}/jax_memory_peak_utilization_max"] = max(
                used / cap for used, cap in pairs
            )
            total_limit = sum(cap for _, cap in pairs)
            if total_limit > 0:
                metrics[f"{prefix}/jax_memory_peak_utilization_total"] = (
                    sum(used for used, _ in pairs) / total_limit
                )

    return metrics


def init_wandb(
    args: BaseArgs,
    cfg,
    *,
    project_name: str,
    group: str | None = None,
    entity: str | None = None,
    run_id_suffix: str | None = None,
) -> tuple[bool, int]:
    """Initialise wandb with resume support.

    Returns ``(use_wandb, wandb_resume_step)``.
    """
    use_wandb = args.use_wandb and is_primary_host()
    wandb_resume_step = 0
    if not use_wandb:
        return use_wandb, wandb_resume_step

    raw_display_name = args.wandb_name or args.experiment
    display_name = _short_wandb_text(raw_display_name)
    explicit_run_id = getattr(args, "wandb_run_id", None)
    if explicit_run_id is not None:
        raw_run_id = explicit_run_id
    else:
        raw_id_name = (
            raw_display_name
            if run_id_suffix is None
            else f"{raw_display_name}-{run_id_suffix}"
        )
        commit, dirty = git_commit_hash()
        if commit is not None:
            if dirty:
                logging.warning(
                    "Git working tree is dirty; wandb run ID uses the last committed state"
                )
            raw_run_id = f"{raw_id_name}-{commit}"
        else:
            raw_run_id = raw_id_name
    run_id = _short_wandb_text(raw_run_id)
    if display_name != raw_display_name or run_id != raw_run_id:
        logging.warning(
            "Shortened wandb name/id to satisfy the %d-character API limit.",
            _WANDB_TEXT_LIMIT,
        )

    init_kwargs = {
        "project": project_name,
        "name": display_name,
        "id": run_id,
        "group": group,
        "resume": "allow",
        "config": asdict(cfg),
    }
    if entity is not None:
        init_kwargs["entity"] = entity

    wandb.init(**init_kwargs)

    if wandb.run.resumed:
        wandb_resume_step = wandb.run.summary.get("step", 0)
        logging.info(f"Resumed wandb run at step {wandb_resume_step}")
    return use_wandb, wandb_resume_step


# ---------------------------------------------------------------------------
# Weight-decay mask
# ---------------------------------------------------------------------------


def build_wd_mask(
    trainable_state,
    *,
    exclude_names: set[str] = frozenset({"pos_embed", "cls_token", "reg_tokens"}),
):
    """Build a weight-decay mask pytree.

    Only applies WD to 2D params (weight matrices), excluding params whose name
    appears in *exclude_names*.
    """

    def wd_mask_fn(path: str, param: nnx.Variable) -> bool:
        if path in exclude_names:
            return False
        if param[...].ndim != 2:
            return False
        return True

    return nnx.map_state(wd_mask_fn, trainable_state)


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


def build_optimizer(
    trainable_pytree,
    trainable_state,
    lr_sched,
    *,
    adam_b1: float,
    adam_b2: float,
    weight_decay: float | Callable[[int], float] = 0.0,
    grad_clip_norm: float = 3.0,
    grad_acc_steps: int = 1,
    wd_exclude_names: set[str] = frozenset({"pos_embed", "cls_token", "reg_tokens"}),
) -> nnx.Optimizer:
    """Build the full optax chain (clip + adamw + MultiSteps) wrapped in nnx.Optimizer.

    When *weight_decay* is callable (a schedule), the optimizer is wrapped with
    ``optax.inject_hyperparams`` so the weight decay value is updated each step.
    """
    wd_mask = build_wd_mask(trainable_state, exclude_names=wd_exclude_names)
    if callable(weight_decay):
        adamw = optax.inject_hyperparams(optax.adamw)(
            lr_sched,
            adam_b1,
            adam_b2,
            weight_decay=weight_decay,
            mask=wd_mask,
        )
    else:
        adamw = optax.adamw(
            lr_sched,
            adam_b1,
            adam_b2,
            weight_decay=weight_decay,
            mask=wd_mask,
        )
    chain = optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw)
    chain = optax.MultiSteps(chain, grad_acc_steps)
    return nnx.Optimizer(trainable_pytree, chain, wrt=nnx.Param)  # ty: ignore


# ---------------------------------------------------------------------------
# Checkpoint manager creation
# ---------------------------------------------------------------------------


def create_checkpoint_manager(
    save_path: Path | str | None,
    item_names: list[str],
    *,
    save_interval_steps: int,
    total_steps: int,
    best_fn=None,
    best_mode: str = "min",
    best_n: int | None = None,
    keep_without_metrics: bool = True,
    max_to_keep: int = 2,
    keep_steps: frozenset[int] | None = None,
) -> ocp.CheckpointManager | None:
    """Create a single checkpoint manager.

    Retention modes:
      * ``keep_steps`` or ``best_n`` set → ``LatestN(max_to_keep)`` unioned with
        ``CustomSteps(keep_steps)`` and/or ``BestN(n=best_n)``. ``BestN`` only
        scores saves that carry a ``metrics`` dict.
      * Otherwise → legacy orbax retention; ``best_fn`` (if any) drives best-N
        with ``max_to_keep`` as the budget (no rolling latest).
    """
    if save_path is None:
        return None
    if isinstance(save_path, Path):
        path_str = str(save_path.absolute())
    else:
        path_str = save_path

    save_on = frozenset([total_steps]) | (keep_steps or frozenset())

    if keep_steps or best_n is not None:
        from orbax.checkpoint._src.checkpoint_managers import preservation_policy as pp

        policies: list[pp.PreservationPolicy] = [pp.LatestN(n=max_to_keep)]
        if keep_steps:
            policies.append(pp.CustomSteps(steps=list(keep_steps)))
        if best_n is not None and best_fn is not None:
            policies.append(
                pp.BestN(
                    get_metric_fn=best_fn,
                    reverse=(best_mode == "min"),
                    n=best_n,
                    keep_checkpoints_without_metrics=False,
                )
            )
        preservation_policy = pp.AnyPreservationPolicy(policies=policies)

        # best_fn must be set so orbax persists per-step metrics to disk (else best_step() can't recover them).
        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            save_on_steps=save_on,
            create=True,
            read_only=False,
            best_fn=best_fn,
            best_mode=best_mode,
            keep_checkpoints_without_metrics=keep_without_metrics,
            preservation_policy=preservation_policy,
        )
    else:
        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            save_on_steps=save_on,
            max_to_keep=max_to_keep,
            create=True,
            read_only=False,
            best_fn=best_fn,
            best_mode=best_mode,
            keep_checkpoints_without_metrics=keep_without_metrics,
        )
    return ocp.CheckpointManager(path_str, options=opts, item_names=item_names)


# ---------------------------------------------------------------------------
# Generic save_checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    manager: ocp.CheckpointManager | None,
    step: int,
    model,
    optim: nnx.Optimizer,
    data_iter,
    cfg,
    metrics: dict | None = None,
    *,
    extra_modules: dict[str, nnx.Module] | None = None,
    extra_optims: dict[str, nnx.Optimizer] | None = None,
) -> bool:
    """Save a checkpoint.  *model* must expose ``get_state() -> dict[str, pytree]``.

    ``extra_modules`` / ``extra_optims`` are additional nnx.Module / nnx.Optimizer
    instances saved as extra named items in the composite (e.g. monitoring probes
    and their optimizers).
    """
    if manager is None:
        return False
    extras = {**(extra_modules or {}), **(extra_optims or {})}
    composite = {
        "optim": ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim))),
        "loader": grain.PyGrainCheckpointSave(data_iter),
        "config": ocp.args.JsonSave(asdict(cfg)),
        **{name: ocp.args.PyTreeSave(tree) for name, tree in model.get_state().items()},
        **{name: ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(m))) for name, m in extras.items()},
    }
    return manager.save(step, metrics=metrics, args=ocp.args.Composite(**composite))


# ---------------------------------------------------------------------------
# Progress bar + profiler setup
# ---------------------------------------------------------------------------


def setup_progress(
    args: BaseArgs,
    optim: nnx.Optimizer,
    total_updates: int,
) -> tuple[tqdm, TrainingProfiler, int]:
    """Create pbar + profiler and read ``updates_completed`` from optimizer state.

    Returns ``(pbar, profiler, updates_completed)``.
    """
    updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
    pbar = tqdm(
        desc="Update",
        initial=updates_completed,
        total=total_updates,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        disable=not is_primary_host() or not sys.stderr.isatty(),
    )
    profiler = TrainingProfiler(
        mode=args.profile_mode,
        port=args.profiler_port,
        start_step=args.profiler_start_step,
        stop_step=args.profiler_stop_step,
    )
    return pbar, profiler, updates_completed


# ---------------------------------------------------------------------------
# Learning-rate schedules shared by the training entry points.
# ---------------------------------------------------------------------------


def build_lr_schedule(
    cfg,
    total_updates: int,
    *,
    schedule: str | None = None,
    warmup_steps_override: int | None = None,
    decay_steps_override: int | None = None,
):
    """Build an Optax LR schedule keyed on optimizer update counts.

    Args:
        cfg: Any object with lr_schedule, epochs, warmup_epochs, lr_start, lr_peak,
            lr_final attributes (duck-typed).
        total_updates: total optimizer steps (after grad accumulation).
        schedule: optional override for cfg.lr_schedule.
        warmup_steps_override: optional warmup step count.
        decay_steps_override: optional decay step count.
    """
    sched = schedule or cfg.lr_schedule
    steps_per_epoch = total_updates // cfg.epochs
    warmup_steps = (
        cfg.warmup_epochs * steps_per_epoch
        if warmup_steps_override is None
        else warmup_steps_override
    )
    decay_steps = decay_steps_override or total_updates

    match sched:
        case "constant":
            return optax.schedules.constant_schedule(cfg.lr_peak)
        case "warmup_cosine":
            return optax.schedules.warmup_cosine_decay_schedule(
                init_value=cfg.lr_start,
                peak_value=cfg.lr_peak,
                warmup_steps=warmup_steps,
                decay_steps=total_updates,
                end_value=cfg.lr_final,
            )
        case "wsd":
            remaining_steps = total_updates - warmup_steps
            if decay_steps_override is not None:
                assert 0 <= decay_steps_override <= remaining_steps, (
                    "decay_steps_override must be between 0 and total_updates - warmup_steps"
                )
            decay_steps = decay_steps_override or remaining_steps
            decay_start = total_updates - decay_steps
            return optax.schedules.join_schedules(
                [
                    optax.schedules.warmup_constant_schedule(
                        init_value=cfg.lr_start, peak_value=cfg.lr_peak, warmup_steps=warmup_steps
                    ),
                    optax.schedules.linear_schedule(
                        init_value=cfg.lr_peak, end_value=cfg.lr_final, transition_steps=decay_steps
                    ),
                ],
                [decay_start],
            )
        case "linear_decay":
            decay_end_epoch = getattr(cfg, "decay_end_epoch", None)
            decay_end_steps = (
                total_updates
                if decay_end_epoch is None
                else min(total_updates, decay_end_epoch * steps_per_epoch)
            )
            if decay_end_steps < warmup_steps:
                raise ValueError(
                    "decay_end_epoch must not precede warmup_epochs for linear_decay."
                )
            decay = max(decay_end_steps - warmup_steps, 1)
            warmup_sched = (
                optax.schedules.linear_schedule(cfg.lr_start, cfg.lr_peak, warmup_steps)
                if warmup_steps > 0 and cfg.lr_start != cfg.lr_peak
                else optax.schedules.constant_schedule(cfg.lr_peak)
            )
            decay_sched = optax.schedules.linear_schedule(cfg.lr_peak, cfg.lr_final, decay)
            schedule = (
                optax.schedules.join_schedules([warmup_sched, decay_sched], [warmup_steps])
                if warmup_steps > 0
                else decay_sched
            )
            if decay_end_steps < total_updates:
                schedule = optax.schedules.join_schedules(
                    [schedule, optax.schedules.constant_schedule(cfg.lr_final)],
                    [decay_end_steps],
                )
            return schedule
        case _:
            raise ValueError(f"Unsupported lr_schedule: {sched}")
