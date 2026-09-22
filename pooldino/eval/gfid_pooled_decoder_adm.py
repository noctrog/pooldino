"""Generate samples from an RAEv2 pooled-latent generator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import jmp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image
import tensorflow_datasets as tfds
import tyro
from absl import logging
from tqdm import tqdm

from pooldino.data.data import _RAEv2HfImageDataSource
from pooldino.guidance import (
    apply_internal_guidance,
    apply_repa_guidance,
)
from pooldino.pooled_generator import (
    decode_pooled_decoder_latents,
    load_pooled_latent_stats,
    restore_pooled_decoder_components,
    restore_pooled_generator,
    validate_pooled_generator_artifacts,
)

PACKAGED_CONDITION_LABELS = "imagenet2012-validation-labels-tfds.npz"


@dataclass
class Config:
    generator_path: Path
    pooled_decoder_path: Path
    output_dir: Path | None = None
    steps: int = 50
    per_class: int = 50
    batch_size: int = 256
    seed: int = 0
    use_ema: bool = True
    use_ema_pooled_decoder: bool = True
    generator_step: int | None = None
    pooled_decoder_step: int | None = None
    implementation: str = "xla"
    precision: Literal["auto", "fp32", "bf16"] = "auto"
    """Auto uses released bf16 inference for exact DDT/source protocols."""
    protocol: Literal[
        "custom",
        "raev2_train",
        "raev2_cfg_sweep",
        "raev2_cfg_sweep_100",
        "raev2_self_repa",
        "raev2_self_repa_100",
        "raev2_ig",
    ] = "custom"
    """Named RAEv2 protocol; the CFG-sweep profile preserves only CFG fields."""
    metric_backend: Literal["adm", "raev2_fd"] = "adm"
    """ADM keeps the legacy external evaluator; raev2_fd runs official fd_evaluator."""
    fd_metrics: tuple[str, ...] = ("fid",)
    fd_metric_batch_size: int = 128
    fd_reference: Path | None = None
    fd_reference_images: Path | None = None
    fd_data_dir: Path | None = None
    fd_device: Literal["auto", "cpu", "cuda"] = "auto"
    uint8_conversion: Literal["auto", "legacy_round", "raev2_truncate"] = "auto"
    condition_labels_path: Path | None = None
    """Packaged official validation-label order; defaults beside decoder runs."""

    cfg_scale: float | None = None
    cfg_t_min: float = 0.3
    cfg_t_max: float = 1.0

    self_repa_guidance_scale: float | None = None
    self_repa_t_min: float = 0.0
    self_repa_t_max: float = 1.0

    # Official RAEv2 convention: base + scale * (full - base), neutral at 1.
    # RAEv2 integrates data->noise time in reverse; its [0.1, 1.0] window maps
    # to the inclusive [0.0, 0.9] interval in this evaluator's noise->data time.
    ig_scale: float | None = None
    ig_t_min: float = 0.0
    ig_t_max: float = 0.9

    stats_path: Path | None = None
    save_npz: bool = True
    save_png: bool = False
    generate_only: bool = False
    """Generate the protocol-exact sample NPZ without running its metric backend."""


def resolve_sampling_protocol(cfg: Config) -> Config:
    """Resolve released ImageNet sampling settings and the CFG-only variant."""

    def guided_cfg() -> tuple[float, float, float]:
        """Keep an explicit CFG override, otherwise use neutral full-window CFG."""

        if cfg.cfg_scale is None:
            return 1.0, 0.0, 1.0
        if cfg.cfg_scale < 1.0:
            raise ValueError(
                "Guided RAEv2 protocols require an explicit --cfg-scale >= 1.0."
            )
        return cfg.cfg_scale, cfg.cfg_t_min, cfg.cfg_t_max

    match cfg.protocol:
        case "custom":
            return cfg
        case "raev2_train":
            return replace(
                cfg,
                steps=50,
                per_class=50,
                seed=42,
                metric_backend="raev2_fd",
                fd_metrics=("fid", "inception_score"),
                use_ema=True,
                use_ema_pooled_decoder=True,
                cfg_scale=1.0,
                cfg_t_min=0.0,
                cfg_t_max=1.0,
                self_repa_guidance_scale=None,
                ig_scale=None,
            )
        case "raev2_cfg_sweep" | "raev2_cfg_sweep_100":
            if cfg.cfg_scale is None or cfg.cfg_scale < 1.0:
                raise ValueError(
                    f"The {cfg.protocol} protocol requires --cfg-scale >= 1.0."
                )
            return replace(
                cfg,
                steps=100 if cfg.protocol == "raev2_cfg_sweep_100" else 50,
                per_class=50,
                seed=42,
                metric_backend="raev2_fd",
                fd_metrics=("fid", "inception_score"),
                use_ema=True,
                use_ema_pooled_decoder=True,
                self_repa_guidance_scale=None,
                ig_scale=None,
            )
        case "raev2_self_repa" | "raev2_self_repa_100":
            if cfg.self_repa_guidance_scale is None:
                raise ValueError(
                    f"The {cfg.protocol} protocol requires "
                    "--self-repa-guidance-scale."
                )
            cfg_scale, cfg_t_min, cfg_t_max = guided_cfg()
            return replace(
                cfg,
                # Preserve the historical 50-step namespace while exposing a
                # separate matched-100-step profile for paper comparisons.
                steps=100 if cfg.protocol == "raev2_self_repa_100" else 50,
                per_class=50,
                seed=42,
                metric_backend="raev2_fd",
                fd_metrics=("fid", "inception_score"),
                use_ema=True,
                use_ema_pooled_decoder=True,
                cfg_scale=cfg_scale,
                cfg_t_min=cfg_t_min,
                cfg_t_max=cfg_t_max,
                ig_scale=None,
            )
        case "raev2_ig":
            cfg_scale, cfg_t_min, cfg_t_max = guided_cfg()
            ig_scale = 1.78 if cfg.ig_scale is None else cfg.ig_scale
            return replace(
                cfg,
                steps=100,
                per_class=50,
                seed=42,
                metric_backend="raev2_fd",
                fd_metrics=("fid", "fdr6", "mind6"),
                use_ema=True,
                use_ema_pooled_decoder=True,
                cfg_scale=cfg_scale,
                cfg_t_min=cfg_t_min,
                cfg_t_max=cfg_t_max,
                self_repa_guidance_scale=None,
                ig_scale=ig_scale,
                # Official data->noise [0.10, 1.0] becomes local
                # noise->data [0.0, 0.90].
                ig_t_min=0.0,
                ig_t_max=0.9,
            )
        case _:
            raise ValueError(f"Unknown sampling protocol: {cfg.protocol!r}.")


def resolve_eval_precision(
    precision: Literal["auto", "fp32", "bf16"],
) -> Literal["fp32", "bf16"]:
    """Resolve the effective inference dtype from an explicit/source profile."""

    if precision == "auto":
        return "bf16"
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported evaluation precision: {precision!r}.")
    return precision

def _output_dir(
    cfg: Config,
    *,
    generator_step: int | None = None,
    pooled_decoder_step: int | None = None,
) -> Path:
    if cfg.output_dir is not None:
        return cfg.output_dir
    parts = [f"samples_s{cfg.steps}"]
    if generator_step is not None:
        parts.append(f"step{generator_step}")
    if pooled_decoder_step is not None:
        parts.append(f"dec{pooled_decoder_step}")
    parts.extend((f"seed{cfg.seed}", "ema" if cfg.use_ema else "raw"))
    if cfg.cfg_scale is None:
        parts.append("nocfg")
    else:
        parts.append(f"cfg{cfg.cfg_scale}")
        parts.append(f"cfgt{cfg.cfg_t_min}-{cfg.cfg_t_max}")
    if cfg.self_repa_guidance_scale is not None:
        parts.append(f"selfrepa{cfg.self_repa_guidance_scale}")
        parts.append(f"srt{cfg.self_repa_t_min}-{cfg.self_repa_t_max}")
    if cfg.ig_scale is not None:
        parts.append(f"ig{cfg.ig_scale}")
        parts.append(f"igt{cfg.ig_t_min}-{cfg.ig_t_max}")
    return cfg.generator_path / "_".join(parts)


def _uint8(images: np.ndarray, *, truncate: bool) -> np.ndarray:
    scaled = np.clip(images, 0.0, 1.0) * 255.0
    if not truncate:
        scaled = scaled + 0.5
    return np.clip(scaled, 0, 255).astype(np.uint8)


def permute_raev2_metric_samples(images: np.ndarray) -> np.ndarray:
    """Apply the released fixed shuffle before distributional evaluation."""

    permutation = np.random.default_rng(0).permutation(images.shape[0])
    return images[permutation]


def resolve_generation_labels(
    cfg: Config,
    *,
    data_cfg,
    num_classes: int,
    condition_labels_path: Path | None = None,
) -> tuple[np.ndarray, str]:
    """Use the configured validation backend's condition order."""

    total_samples = num_classes * cfg.per_class
    if cfg.protocol == "custom":
        return (
            np.repeat(np.arange(num_classes, dtype=np.int32), cfg.per_class),
            "balanced_class_repeat",
        )
    if num_classes != 1000 or cfg.per_class != 50:
        raise ValueError(
            "Released RAEv2 sampling protocols require exactly 1000 classes "
            "and 50 validation conditions per class."
        )
    source_length: int
    if condition_labels_path is not None:
        condition_labels_path = Path(condition_labels_path)
        if not condition_labels_path.is_file():
            raise FileNotFoundError(
                f"Packaged condition labels not found: {condition_labels_path}"
            )
        with np.load(condition_labels_path, allow_pickle=False) as archive:
            required = {
                "format_version",
                "dataset",
                "split",
                "labels",
                "labels_sha256",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    "Packaged condition-label file is missing metadata: "
                    f"{sorted(missing)}."
                )
            if int(np.asarray(archive["format_version"]).item()) != 1:
                raise ValueError("Unsupported packaged condition-label format.")
            if str(np.asarray(archive["dataset"]).item()) != data_cfg.dataset:
                raise ValueError("Packaged condition-label dataset does not match.")
            if str(np.asarray(archive["split"]).item()) != data_cfg.val_name:
                raise ValueError("Packaged condition-label split does not match.")
            labels = np.asarray(archive["labels"])
            expected_sha256 = str(np.asarray(archive["labels_sha256"]).item())
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Packaged condition labels must have an integer dtype.")
        actual_sha256 = hashlib.sha256(
            np.ascontiguousarray(labels).tobytes()
        ).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Packaged condition-label checksum does not match.")
        labels = labels.astype(np.int32, copy=False)
        source_length = len(labels)
        provenance = f"packaged_tfds_validation_order:{condition_labels_path}"
    elif data_cfg.backend == "raev2_hf" and data_cfg.data_dir is not None:
        source = _RAEv2HfImageDataSource(data_cfg.data_dir, data_cfg.val_name)
        labels = source.labels()
        source_length = len(source)
        provenance = "raev2_hf_validation_arrow_order"
    elif data_cfg.backend == "tfds":
        source = tfds.data_source(
            data_cfg.dataset,
            split=data_cfg.val_name,
            decoders={"image": tfds.decode.SkipDecoding()},
        )
        labels = np.fromiter(
            (int(source[index]["label"]) for index in range(len(source))),
            dtype=np.int32,
            count=len(source),
        )
        source_length = len(source)
        provenance = "tfds_validation_order"
    else:
        raise ValueError(
            f"Sampling protocol {cfg.protocol!r} requires either the RAEv2 "
            "Arrow validation data or ImageNet TFDS."
        )
    if source_length != total_samples:
        raise ValueError(
            f"ImageNet validation split must contain exactly {total_samples} "
            f"labels, got {source_length}."
        )
    if labels.shape != (total_samples,):
        raise ValueError(
            f"Expected {total_samples} validation labels, got {labels.shape}."
        )
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            f"Validation labels must lie in [0, {num_classes}), got "
            f"range [{labels.min()}, {labels.max()}]."
        )
    class_counts = np.bincount(labels, minlength=num_classes)
    if not np.array_equal(class_counts, np.full(num_classes, cfg.per_class)):
        raise ValueError(
            "ImageNet validation labels must contain exactly 50 examples "
            "from each of the 1000 ImageNet classes."
        )
    return labels, provenance


def resolve_condition_labels_path(cfg: Config) -> Path | None:
    """Prefer migrated label metadata and retain TFDS as a compatibility fallback."""

    if cfg.condition_labels_path is not None:
        return cfg.condition_labels_path
    candidate = cfg.pooled_decoder_path.parent / PACKAGED_CONDITION_LABELS
    return candidate if candidate.is_file() else None


def resolve_fd_reference_images(
    metrics: tuple[str, ...],
    *,
    reference_images: Path | None,
    data_dir: Path | str | None,
) -> Path | None:
    """Resolve MIND images in the same order as GitHub RAEv2."""

    needs_mind = any(name.startswith("mind") for name in metrics)
    if reference_images is not None:
        return Path(reference_images)
    env_reference = os.environ.get("NANOGEN_EVALS_REF_IMAGES")
    if env_reference:
        return Path(env_reference)
    if needs_mind and data_dir is not None:
        candidate = Path(data_dir).expanduser() / "imagenet-256-val.npz"
        if candidate.exists():
            return candidate
    if needs_mind:
        raise FileNotFoundError(
            "MIND metrics require raw ImageNet validation images. Pass "
            "--fd-reference-images, set NANOGEN_EVALS_REF_IMAGES, or place "
            "imagenet-256-val.npz under the stage-one data directory."
        )
    return None


def require_raev2_fd_evaluator():
    """Import the released metric backend, failing before expensive sampling."""

    try:
        import torch
        from fd_evaluator import compute_metrics
    except ImportError as error:
        raise RuntimeError(
            "metric_backend='raev2_fd' requires the fd_evaluator package used "
            "by nanovisionx/RAEv2. Install it in the evaluation environment."
        ) from error
    return torch, compute_metrics


def compute_raev2_fd_metrics(
    images: np.ndarray,
    *,
    metrics: tuple[str, ...] = ("fid",),
    reference: Path | None = None,
    reference_images: Path | None = None,
    data_dir: Path | str | None = None,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    batch_size: int = 128,
) -> dict[str, float]:
    """Delegate to the fd_evaluator call used by released RAEv2."""

    torch, compute_metrics = require_raev2_fd_evaluator()

    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("fd_evaluator images must be uint8 NHWC RGB.")
    if not metrics:
        raise ValueError("fd_metrics must contain at least one metric.")
    if batch_size <= 0:
        raise ValueError("fd_metric_batch_size must be positive.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    reference_images = resolve_fd_reference_images(
        metrics,
        reference_images=reference_images,
        data_dir=data_dir,
    )

    cache_dir = os.environ.get(
        "NANOGEN_EVALS_CACHE_DIR",
        str(Path.home() / ".cache" / "nanogen-evals" / "features"),
    )
    kwargs = dict(
        images=images,
        metrics=list(metrics),
        reference_images=(str(reference_images) if reference_images else None),
        device=device,
        batch_size=batch_size,
        feature_cache_dir=cache_dir,
        feature_cache_key=None,
        reference_feature_cache_key="imagenet256_val",
        verbose=True,
    )
    if reference is not None:
        kwargs["fid_reference"] = str(reference)
    return {name: float(value) for name, value in compute_metrics(**kwargs).items()}


def _load_stats(
    path: Path,
    eps: float,
    *,
    restored,
):
    mean_np, var_np, identity = load_pooled_latent_stats(
        path,
        expected_shape=(
            restored.latent_spec.num_latents,
            restored.latent_spec.feat,
        ),
        expected_decoder_identity=restored.identity,
        require_source_metadata=True,
    )
    mean = jnp.asarray(mean_np, dtype=jnp.float32)
    var = jnp.asarray(var_np, dtype=jnp.float32)
    std = jnp.sqrt(var + eps)
    return mean, std, identity


def _time_shift(generator_cfg, num_latents: int, feat: int) -> float | None:
    if generator_cfg.kappa is not None:
        return generator_cfg.kappa
    if generator_cfg.time_dist_shift_base is None:
        return None
    shift_dim = generator_cfg.time_dist_shift_dim
    if shift_dim is None:
        shift_dim = num_latents * feat
    return max(1.0, (shift_dim / generator_cfg.time_dist_shift_base) ** 0.5)


def _time_in_interval(
    value: jax.Array,
    interval: tuple[float, float],
    *,
    inclusive: bool,
) -> bool:
    """Compare a schedule value with endpoints represented in its own dtype."""
    lower = jnp.asarray(interval[0], dtype=value.dtype)
    upper = jnp.asarray(interval[1], dtype=value.dtype)
    if inclusive:
        # ``linspace`` can place nominal decimal endpoints one ULP outside the
        # independently rounded Python value (notably 0.9 in float32).
        lower = jnp.nextafter(lower, jnp.asarray(-jnp.inf, dtype=value.dtype))
        upper = jnp.nextafter(upper, jnp.asarray(jnp.inf, dtype=value.dtype))
        selected = (lower <= value) & (value <= upper)
    else:
        selected = (lower < value) & (value < upper)
    return bool(jax.device_get(selected))


def _sampling_time_coordinates(
    steps: int,
    dtype,
    *,
    time_shift: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Build local integration and exact model-time Euler coordinates.

    For exact RAEv2 models this first reproduces the released decreasing
    official grid and its shift, then complements it only for local stepping.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if time_shift is not None and time_shift <= 0:
        raise ValueError("time_shift must be positive when provided.")
    official_t = jnp.linspace(1.0, 0.0, steps + 1, dtype=dtype)
    if time_shift is not None:
        shift = jnp.asarray(time_shift, dtype=dtype)
        official_t = shift * official_t / (
            1.0 + (shift - 1.0) * official_t
        )
    return 1.0 - official_t, official_t


def _sampling_time_grid(
    steps: int,
    dtype,
    *,
    time_shift: float | None,
) -> jax.Array:
    """Compatibility helper returning the local increasing integration grid."""

    local_t, _ = _sampling_time_coordinates(
        steps,
        dtype,
        time_shift=time_shift,
    )
    return local_t


@nnx.jit(
    static_argnames=(
        "xpred_denom_eps",
        "use_cfg",
        "use_self_repa",
        "use_internal_guidance",
        "self_repa_source_grid",
    )
)
def integration_step(
    generator,
    state: jax.Array,
    t: jax.Array,
    dt: jax.Array,
    labels: jax.Array,
    null_labels: jax.Array,
    *,
    use_cfg: bool,
    cfg_scale: float,
    use_self_repa: bool,
    self_repa_scale: float,
    self_repa_tokenizer=None,
    self_repa_source_grid: tuple[int, int] | None = None,
    use_internal_guidance: bool = False,
    ig_scale: float = 1.0,
    xpred_denom_eps: float = 0.05,
) -> jax.Array:
    if use_self_repa and use_internal_guidance:
        raise ValueError("Self-REPA guidance and internal guidance are mutually exclusive.")

    def velocity_for(current_labels: jax.Array) -> jax.Array:
        output = generator(
            state,
            t,
            current_labels,
            train=False,
            return_self_repa=use_self_repa,
            return_base_model=use_internal_guidance,
        )
        clean = output["x"]
        if use_self_repa:
            self_repa_clean = output["self_repa"]
            if self_repa_tokenizer is not None:
                if self_repa_source_grid is None:
                    raise ValueError(
                        "self_repa_source_grid is required with a self-REPA tokenizer."
                    )
                self_repa_clean = self_repa_tokenizer(
                    self_repa_clean,
                    grid_hw=self_repa_source_grid,
                )
            if self_repa_clean.shape != clean.shape:
                raise ValueError(
                    "The tokenized self-REPA prediction must match the generated "
                    f"latent shape; got {self_repa_clean.shape} and {clean.shape}."
                )
            self_repa_clean = generator.normalize(self_repa_clean)
            clean = apply_repa_guidance(clean, self_repa_clean, self_repa_scale)
        if use_internal_guidance:
            clean = apply_internal_guidance(clean, output["base_x"], ig_scale)
        model_t_broadcast = t.reshape(
            (t.shape[0],) + (1,) * (state.ndim - 1)
        )
        denom = jnp.maximum(
            model_t_broadcast,
            jnp.asarray(xpred_denom_eps, dtype=t.dtype),
        )
        # Released sampler drift: (x_t - x_pred) / clamp(t_official).
        return (state - clean) / denom

    conditional_velocity = velocity_for(labels)
    if use_cfg:
        unconditional_velocity = velocity_for(null_labels)
        velocity = unconditional_velocity + cfg_scale * (
            conditional_velocity - unconditional_velocity
        )
    else:
        velocity = conditional_velocity
    # ``dt`` is the positive decrement in official time. This is the released
    # Euler update ``x = x - h * drift``.
    return state - velocity * dt


def sample_latents(
    generator,
    labels: jax.Array,
    key: jax.Array,
    *,
    latent_shape: tuple[int, int],
    steps: int,
    time_shift: float | None,
    cfg_scale: float | None,
    cfg_interval: tuple[float, float],
    self_repa_scale: float | None,
    self_repa_interval: tuple[float, float],
    self_repa_tokenizer=None,
    self_repa_source_grid: tuple[int, int] | None = None,
    ig_scale: float | None = None,
    ig_interval: tuple[float, float] = (0.0, 0.9),
    xpred_denom_eps: float = 0.05,
) -> jax.Array:
    if self_repa_scale is not None and ig_scale is not None:
        raise ValueError("Self-REPA guidance and internal guidance are mutually exclusive.")
    state = jax.random.normal(
        key,
        (labels.shape[0],) + latent_shape,
        dtype=jnp.float32,
    )
    ts, model_ts = _sampling_time_coordinates(
        steps,
        state.dtype,
        time_shift=time_shift,
    )
    dts = ts[1:] - ts[:-1]
    null_labels = jnp.full_like(labels, generator.cfg.num_classes)

    for index in range(steps):
        t_value = ts[index]
        use_cfg = (
            cfg_scale is not None
            # Released GuidanceConfig activates CFG only above its neutral
            # scale of one. Both published ImageNet protocols set scale=1.
            and cfg_scale > 1.0
            and _time_in_interval(t_value, cfg_interval, inclusive=False)
        )
        use_self_repa = (
            self_repa_scale is not None
            and _time_in_interval(t_value, self_repa_interval, inclusive=False)
        )
        # Inclusive bounds match the official RAEv2 schedule after complementing
        # its decreasing-time interval into this evaluator's increasing time.
        use_internal_guidance = (
            ig_scale is not None
            and _time_in_interval(t_value, ig_interval, inclusive=True)
        )
        # Keep the released shifted official coordinate instead of recovering
        # it with ``1 - (1 - t)`` after the local-orientation conversion.
        t = jnp.full((labels.shape[0],), model_ts[index], dtype=state.dtype)
        state = integration_step(
            generator,
            state,
            t,
            dts[index],
            labels,
            null_labels,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale or 0.0,
            use_self_repa=use_self_repa,
            self_repa_scale=self_repa_scale or 0.0,
            self_repa_tokenizer=self_repa_tokenizer,
            self_repa_source_grid=self_repa_source_grid,
            use_internal_guidance=use_internal_guidance,
            ig_scale=ig_scale if ig_scale is not None else 1.0,
            xpred_denom_eps=xpred_denom_eps,
        )
    return state


def main(cfg: Config) -> None:
    cfg = resolve_sampling_protocol(cfg)
    if cfg.steps <= 0 or cfg.batch_size <= 0 or cfg.per_class <= 0:
        raise ValueError("steps, batch_size, and per_class must be positive.")
    if not 0 <= cfg.cfg_t_min < cfg.cfg_t_max <= 1:
        raise ValueError("Invalid CFG interval.")
    if not 0 <= cfg.self_repa_t_min < cfg.self_repa_t_max <= 1:
        raise ValueError("Invalid self-REPA guidance interval.")
    if not 0 <= cfg.ig_t_min < cfg.ig_t_max <= 1:
        raise ValueError("Invalid internal-guidance interval.")
    if cfg.self_repa_guidance_scale is not None and cfg.ig_scale is not None:
        raise ValueError(
            "--self-repa-guidance-scale and --ig-scale use different conventions and "
            "cannot be combined."
        )
    if (
        cfg.self_repa_guidance_scale is not None
        and cfg.self_repa_guidance_scale < 0
    ):
        raise ValueError("Self-REPA guidance scale must be non-negative.")
    if cfg.ig_scale is not None and cfg.ig_scale < 0:
        raise ValueError("Internal-guidance scale must be non-negative.")
    if cfg.metric_backend not in ("adm", "raev2_fd"):
        raise ValueError(f"Unsupported metric backend: {cfg.metric_backend!r}.")
    if cfg.metric_backend == "raev2_fd" and not cfg.generate_only:
        require_raev2_fd_evaluator()
    if cfg.uint8_conversion not in ("auto", "legacy_round", "raev2_truncate"):
        raise ValueError(f"Unsupported uint8 conversion: {cfg.uint8_conversion!r}.")

    manager = ocp.CheckpointManager(
        cfg.generator_path.absolute(),
        item_names=["model", "model_ema", "optim", "loader", "config"],
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    generator_step = cfg.generator_step or manager.latest_step()
    if generator_step is None:
        raise ValueError(f"No generator checkpoint found at {cfg.generator_path}.")
    raw_cfg = manager.restore(
        generator_step,
        args=ocp.args.Composite(config=ocp.args.JsonRestore()),
    )["config"]
    manager.close()
    if raw_cfg.get("model_type", "raev2_ddt") != "raev2_ddt":
        raise ValueError("This evaluator only restores RAEv2 DDT checkpoints.")
    effective_precision = resolve_eval_precision(
        cfg.precision,
    )

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=(jnp.bfloat16 if effective_precision == "bf16" else jnp.float32),
        output_dtype=jnp.float32,
    )
    use_ema_pooled_decoder = cfg.use_ema_pooled_decoder
    restored = restore_pooled_decoder_components(
        cfg.pooled_decoder_path,
        mesh=mesh,
        mp=mp,
        seed=cfg.seed,
        step=cfg.pooled_decoder_step,
        use_ema=use_ema_pooled_decoder,
        restore_decoder=True,
        implementation=cfg.implementation,
    )
    if restored.decoder is None:
        raise RuntimeError("Pooled decoder was not restored.")

    stats_path = cfg.stats_path or cfg.pooled_decoder_path / "pooled_latent_stats.npz"
    latent_mean, latent_std, stats_identity = _load_stats(
        stats_path,
        raw_cfg.get("latent_norm_eps", 1e-5),
        restored=restored,
    )

    generator, generator_cfg, generator_step = restore_pooled_generator(
        cfg.generator_path,
        latent_spec=restored.latent_spec,
        mesh=mesh,
        mp=mp,
        use_ema=cfg.use_ema,
        step=generator_step,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    validate_pooled_generator_artifacts(
        generator_cfg,
        restored,
        stats_identity,
    )
    if cfg.self_repa_guidance_scale is not None and not generator_cfg.self_repa:
        raise ValueError(
            "Self-REPA guidance requested from a generator without that head."
        )
    if (
        cfg.self_repa_guidance_scale is not None
        and generator_cfg.self_repa_target_grid != restored.grid_hw
    ):
        raise ValueError(
            "The self-REPA target grid must match the pooled decoder's full source "
            f"grid; got {generator_cfg.self_repa_target_grid} and {restored.grid_hw}."
        )
    if cfg.self_repa_guidance_scale is not None and restored.cfg.post_pool_norm:
        raise ValueError(
            "Exact self-REPA projection is unavailable for post-pool-normalized "
            "decoders because their dense target does not retain the separated "
            "final-layer mean. The RAEv2-official decoders do not use this option."
        )

    time_shift = _time_shift(
        generator_cfg,
        restored.latent_spec.num_latents,
        restored.latent_spec.feat,
    )
    output_dir = _output_dir(
        cfg,
        generator_step=generator_step,
        pooled_decoder_step=restored.step,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    truncate_uint8 = (
        cfg.uint8_conversion == "raev2_truncate"
        or cfg.uint8_conversion == "auto"
    )
    resolved_fd_reference_images = None
    resolved_fd_data_dir = (
        cfg.fd_data_dir
        if cfg.fd_data_dir is not None
        else getattr(restored.data_cfg, "data_dir", None)
    )
    if cfg.metric_backend == "raev2_fd" and not cfg.generate_only:
        resolved_fd_reference_images = resolve_fd_reference_images(
            cfg.fd_metrics,
            reference_images=cfg.fd_reference_images,
            data_dir=resolved_fd_data_dir,
        )

    num_classes = generator.cfg.num_classes
    condition_labels_path = resolve_condition_labels_path(cfg)
    labels_all, condition_source = resolve_generation_labels(
        cfg,
        data_cfg=restored.data_cfg,
        num_classes=num_classes,
        condition_labels_path=condition_labels_path,
    )
    total_samples = len(labels_all)
    all_samples: list[np.ndarray] = []
    sharding = NamedSharding(mesh, P("data"))
    key = jax.random.PRNGKey(cfg.seed)

    progress = tqdm(total=total_samples, desc="Generating")
    for start in range(0, total_samples, cfg.batch_size):
        labels_np = labels_all[start : start + cfg.batch_size]
        real_batch = len(labels_np)
        padding = (-real_batch) % mesh.size
        if padding:
            labels_np = np.concatenate((labels_np, np.repeat(labels_np[-1:], padding)))
        labels = jax.device_put(jnp.asarray(labels_np, dtype=jnp.int32), sharding)
        key, sample_key = jax.random.split(key)
        latents = sample_latents(
            generator,
            labels,
            sample_key,
            latent_shape=(restored.latent_spec.num_latents, restored.latent_spec.feat),
            steps=cfg.steps,
            time_shift=time_shift,
            cfg_scale=cfg.cfg_scale,
            cfg_interval=(cfg.cfg_t_min, cfg.cfg_t_max),
            self_repa_scale=cfg.self_repa_guidance_scale,
            self_repa_interval=(cfg.self_repa_t_min, cfg.self_repa_t_max),
            self_repa_tokenizer=(
                restored.tokenizer
                if cfg.self_repa_guidance_scale is not None
                else None
            ),
            self_repa_source_grid=(
                restored.grid_hw
                if cfg.self_repa_guidance_scale is not None
                else None
            ),
            ig_scale=cfg.ig_scale,
            ig_interval=(cfg.ig_t_min, cfg.ig_t_max),
            xpred_denom_eps=generator_cfg.xpred_denom_eps,
        )
        latents = generator.denormalize(latents)
        images = decode_pooled_decoder_latents(
            restored.decoder,
            latents,
            grid_hw=restored.grid_hw,
            pool_hw=restored.cfg.pool_window,
            repeat_pool=restored.cfg.repeat_pool,
            num_output_tokens=restored.num_output_tokens,
        )
        images_np = np.asarray(jax.device_get(images))
        if padding:
            images_np = images_np[:-padding]
        images_uint8 = _uint8(images_np, truncate=truncate_uint8)
        if cfg.save_npz or cfg.metric_backend == "raev2_fd":
            all_samples.append(images_uint8)
        if cfg.save_png:
            for offset, image in enumerate(images_uint8):
                label = int(labels_all[start + offset])
                path = (
                    output_dir
                    / "images"
                    / f"class_{label:04d}"
                    / f"sample_{start + offset:06d}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
        progress.update(real_batch)
    progress.close()

    samples = np.concatenate(all_samples, axis=0) if all_samples else None
    if samples is not None and cfg.metric_backend == "raev2_fd":
        # GitHub RAEv2 shuffles the gathered uint8 array with this fixed seed.
        # This is essential for Inception Score when generation is class-ordered.
        samples = permute_raev2_metric_samples(samples)
    if cfg.save_npz:
        assert samples is not None
        np.savez(output_dir / "samples.npz", arr_0=samples)
    if cfg.metric_backend == "raev2_fd" and not cfg.generate_only:
        assert samples is not None
        fd_results = compute_raev2_fd_metrics(
            samples,
            metrics=cfg.fd_metrics,
            reference=cfg.fd_reference,
            reference_images=resolved_fd_reference_images,
            data_dir=resolved_fd_data_dir,
            device=cfg.fd_device,
            batch_size=cfg.fd_metric_batch_size,
        )
        with open(output_dir / "raev2_fd_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(fd_results, handle, indent=2, sort_keys=True)
    with open(output_dir / "generation_config.txt", "w", encoding="utf-8") as handle:
        handle.write(f"generator_step: {generator_step}\n")
        handle.write(f"pooled_decoder_step: {restored.step}\n")
        handle.write(f"seed: {cfg.seed}\n")
        handle.write(
            f"source_config_global_seed: {42 if cfg.protocol != 'custom' else None}\n"
        )
        handle.write(
            "rng_topology: local_single_jax_stream; GitHub distributed Torch uses "
            "global_seed*world_size+rank and is not bitwise equivalent\n"
        )
        handle.write(f"use_ema: {cfg.use_ema}\n")
        handle.write(f"use_ema_pooled_decoder: {use_ema_pooled_decoder}\n")
        handle.write("model_type: raev2_ddt\n")
        handle.write(f"precision: {effective_precision}\n")
        handle.write(f"sampling_protocol: {cfg.protocol}\n")
        handle.write(f"condition_source: {condition_source}\n")
        handle.write(f"condition_labels_path: {condition_labels_path}\n")
        handle.write(
            "condition_data_dir: "
            f"{getattr(restored.data_cfg, 'data_dir', None) if cfg.protocol != 'custom' else None}\n"
        )
        handle.write(
            "condition_split: "
            f"{getattr(restored.data_cfg, 'val_name', None) if cfg.protocol != 'custom' else None}\n"
        )
        handle.write(f"metric_backend: {cfg.metric_backend}\n")
        handle.write(f"generate_only: {cfg.generate_only}\n")
        handle.write(f"fd_metrics: {cfg.fd_metrics}\n")
        handle.write(f"fd_metric_batch_size: {cfg.fd_metric_batch_size}\n")
        handle.write(f"fd_reference: {cfg.fd_reference}\n")
        handle.write(f"fd_reference_images: {resolved_fd_reference_images}\n")
        handle.write(f"fd_data_dir: {resolved_fd_data_dir}\n")
        handle.write(
            f"fd_sample_permutation_seed: {0 if cfg.metric_backend == 'raev2_fd' else None}\n"
        )
        conversion_name = "raev2_truncate" if truncate_uint8 else "legacy_round"
        handle.write(f"uint8_conversion: {conversion_name}\n")
        handle.write("model_time_orientation: data_to_noise\n")
        handle.write(f"steps: {cfg.steps}\n")
        handle.write(f"cfg_scale: {cfg.cfg_scale}\n")
        handle.write(f"cfg_t_interval_local: {(cfg.cfg_t_min, cfg.cfg_t_max)}\n")
        handle.write(
            f"self_repa_guidance_scale: {cfg.self_repa_guidance_scale}\n"
        )
        handle.write(
            "self_repa_t_interval: "
            f"{(cfg.self_repa_t_min, cfg.self_repa_t_max)}\n"
        )
        handle.write(f"ig_scale_official_convention: {cfg.ig_scale}\n")
        handle.write(f"ig_t_interval_local: {(cfg.ig_t_min, cfg.ig_t_max)}\n")
        handle.write(
            f"ig_t_interval_raev2: {(1.0 - cfg.ig_t_max, 1.0 - cfg.ig_t_min)}\n"
        )
        handle.write("ig_time_orientation: local_noise_0_to_data_1\n")
        handle.write(f"base_model_depth: {generator_cfg.base_model_depth}\n")
        handle.write(f"xpred_denom_eps: {generator_cfg.xpred_denom_eps}\n")
        handle.write("pred_type: x\n")
        handle.write(f"time_shift: {time_shift}\n")
        handle.write(
            f"pooled_decoder_identity: {generator_cfg.pooled_decoder_identity}\n"
        )
        handle.write(f"latent_stats_identity: {generator_cfg.latent_stats_identity}\n")
        handle.write(
            f"latent_shape: {(restored.latent_spec.num_latents, restored.latent_spec.feat)}\n"
        )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(tyro.cli(Config))
