"""Helpers for flow generators trained on pooled-decoder latents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import orbax.checkpoint as ocp
from dacite import Config as DaciteConfig
from dacite import from_dict
from flax import nnx

from pooldino.models.transformer import set_attn_implementation
from pooldino.paths import released_decoder_artifact_identity, remap_artifact_path
from pooldino.generator_config import GeneratorOptimConfig
from pooldino.backbone import load_backbone
from pooldino.decoder_config import ITEM_NAMES
from pooldino.train_decoder import (
    POOLED_ITEM_NAMES,
    PatchDownsampler,
    PooledDecoderConfig,
    extract_pool_source_tokens,
    finish_pooled_tokens,
    pooled_grid_shape,
    repeat_pooled_tokens,
    resolve_representation_layers,
)
from pooldino.models.decoder import RAEDecoder
from pooldino.models.raev2_ddt import RAEv2DDT, RAEv2DDTConfig

RAEV2_BASE_MODEL_DEPTH = 8
RAEV2_BASE_MODEL_COEFF = 1.0
RAEV2_SELF_REPA_LAYER = 8
RAEV2_SELF_REPA_COEFF = 0.5
RAEV2_TRANSPORT_EPS = 0.05
RAEV2_LATENT_NORM_EPS = 1e-5
RAEV2_CFG_DROPOUT_PROB = 0.1
RAEV2_TIME_DIST_SHIFT_DIM = 262_144
RAEV2_REFERENCE_TRAIN_EPOCHS = 80


@dataclass(frozen=True)
class PooledDecoderIdentity:
    """Immutable identity of the stage-one artifact consumed by stage two."""

    checkpoint_path: str
    checkpoint_step: int
    use_ema: bool
    stage1_profile: str
    config_sha256: str


@dataclass(frozen=True)
class PooledLatentStatsIdentity:
    """Identity of the exact latent-statistics file used for normalization."""

    stats_path: str
    sha256: str
    count: int
    shape: tuple[int, ...]


def _raev2_train_config() -> GeneratorOptimConfig:
    return GeneratorOptimConfig(
        epochs=RAEV2_REFERENCE_TRAIN_EPOCHS,
        batch_size=1024,
        lr_start=2e-4,
        lr_peak=2e-4,
        lr_final=2e-5,
        warmup_epochs=25,
        ema=0.9995,
        lr_schedule="linear_decay",
        optimizer="gmuon",
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        decay_end_epoch=50,
    )


@dataclass
class PooledGeneratorConfig:
    """RAEv2 stage-two config, including spatial-pooling research extensions."""

    train: GeneratorOptimConfig = field(default_factory=_raev2_train_config)
    raev2_ddt: RAEv2DDTConfig = field(default_factory=RAEv2DDTConfig)

    # Fixed RAEv2 transport and normalization semantics.
    latent_norm_eps: float = RAEV2_LATENT_NORM_EPS
    time_mu: float = 0.0
    time_sigma: float = 1.0
    time_dist_shift_base: int = 4096
    time_dist_shift_dim: int | None = RAEV2_TIME_DIST_SHIFT_DIM
    kappa: float | None = None
    conditioning_dropout_prob: float = RAEV2_CFG_DROPOUT_PROB

    # The released internal-guidance head remains part of the reference DDT.
    base_model_depth: int = RAEV2_BASE_MODEL_DEPTH
    base_model_coeff: float = RAEV2_BASE_MODEL_COEFF
    xpred_denom_eps: float = RAEV2_TRANSPORT_EPS

    # Reference self-REPA predicts the frozen source-encoder representation at
    # block eight. For compressed grids the head expands each pooled token to
    # its corresponding full-resolution source cell; this is our extension.
    self_repa: bool = True
    self_repa_layer: int = RAEV2_SELF_REPA_LAYER
    self_repa_coeff: float = RAEV2_SELF_REPA_COEFF
    self_repa_cosine_weight: float = 0.0
    self_repa_target_grid: tuple[int, int] = (16, 16)

    # Stage two is only meaningful relative to a particular stage-one
    # tokenizer/decoder and normalization-statistics file.  New checkpoints
    # persist both identities and validate them on resume/evaluation.
    pooled_decoder_identity: PooledDecoderIdentity | None = None
    latent_stats_identity: PooledLatentStatsIdentity | None = None
    optimizer_partition_counts: dict[str, int] | None = None

    def __post_init__(self):
        expected_train = _raev2_train_config()
        if (
            self.train.epochs < RAEV2_REFERENCE_TRAIN_EPOCHS
            or replace(self.train, epochs=RAEV2_REFERENCE_TRAIN_EPOCHS)
            != expected_train
        ):
            raise ValueError(
                "Pooled generator training is fixed to the 80-epoch RAEv2 "
                "optimizer and learning-rate recipe; only its horizon may be extended."
            )
        reference_model = RAEv2DDTConfig(
            input_grid=self.raev2_ddt.input_grid,
            in_channels=self.raev2_ddt.in_channels,
        )
        if self.raev2_ddt != reference_model:
            raise ValueError(
                "Pooled generators use the reference RAEv2 DDT architecture; only "
                "the pooled input grid and channel count may vary."
            )
        if (
            self.latent_norm_eps != RAEV2_LATENT_NORM_EPS
            or self.time_mu != 0.0
            or self.time_sigma != 1.0
            or self.time_dist_shift_base != 4096
            or self.conditioning_dropout_prob != RAEV2_CFG_DROPOUT_PROB
            or self.base_model_depth != RAEV2_BASE_MODEL_DEPTH
            or self.base_model_coeff != RAEV2_BASE_MODEL_COEFF
            or self.xpred_denom_eps != RAEV2_TRANSPORT_EPS
        ):
            raise ValueError(
                "Latent normalization, transport, conditioning dropout, and base "
                "head settings are fixed to the RAEv2 reference recipe."
            )
        if not 1 <= self.self_repa_layer <= self.raev2_ddt.encoder_depth:
            raise ValueError("self_repa_layer must lie within the encoder depth.")
        if self.self_repa_coeff < 0 or self.self_repa_cosine_weight < 0:
            raise ValueError("Self-REPA loss weights must be non-negative.")
        if len(self.self_repa_target_grid) != 2 or min(self.self_repa_target_grid) <= 0:
            raise ValueError("self_repa_target_grid must contain two positive dimensions.")
        if self.time_dist_shift_dim not in (None, RAEV2_TIME_DIST_SHIFT_DIM):
            raise ValueError(
                "time_dist_shift_dim must be the released value or dynamically "
                "derived for a pooling experiment."
            )
        if self.kappa is not None and self.kappa <= 0:
            raise ValueError("kappa must be positive when provided.")
        if self.optimizer_partition_counts is not None:
            expected_keys = {
                "gmuon_tensors",
                "gmuon_parameters",
                "adamw_tensors",
                "adamw_parameters",
            }
            if set(self.optimizer_partition_counts) != expected_keys:
                raise ValueError(
                    "optimizer_partition_counts must contain exactly "
                    f"{sorted(expected_keys)}."
                )
            if any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in self.optimizer_partition_counts.values()
            ):
                raise ValueError(
                    "optimizer_partition_counts values must be non-negative integers."
                )


@dataclass(frozen=True)
class PooledLatentSpec:
    """Shape of the compressed token field used by a pooled decoder."""

    grid: tuple[int, int]
    feat: int

    @property
    def num_latents(self) -> int:
        return self.grid[0] * self.grid[1]


@dataclass(frozen=True)
class RestoredPooledDecoderComponents:
    dino: object
    decoder: RAEDecoder | None
    tokenizer: PatchDownsampler
    cfg: PooledDecoderConfig
    data_cfg: object
    aug_cfg: object
    step: int
    grid_hw: tuple[int, int]
    latent_spec: PooledLatentSpec
    layer_indices: tuple[int, ...]
    num_output_tokens: int
    identity: PooledDecoderIdentity


def _canonical_artifact_path(path: Path | str) -> str:
    """Return a stable local/GCS artifact identifier for checkpoint metadata."""

    path_str = str(path)
    if path_str.startswith("gs://"):
        return path_str.rstrip("/")
    return str(Path(path_str).expanduser().resolve())


def _artifact_path_identity(path: Path | str) -> tuple[str, ...]:
    """Return a relocation-safe identity for a persisted artifact path.

    Released decoder artifacts use their known Hub-relative run identity, so
    downloading them to a different root requires no environment overrides.
    Other local artifacts preserve the existing output-relative convention;
    paths outside either convention, and remote URIs, remain exact matches.
    """

    path_str = remap_artifact_path(path).rstrip("/")
    if "://" in path_str:
        return ("uri", path_str)

    resolved = Path(path_str).expanduser().resolve()
    if released_identity := released_decoder_artifact_identity(resolved):
        return released_identity
    parts = resolved.parts
    output_indices = [index for index, part in enumerate(parts) if part == "output"]
    if output_indices:
        return ("output", *parts[output_indices[-1] + 1 :])
    return ("local", str(resolved))


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_scalar(values, name: str):
    array = np.asarray(values[name])
    if array.shape != ():
        raise ValueError(f"Latent-stats metadata {name!r} must be scalar, got {array.shape}.")
    return array.item()


def _decoder_identity_mismatches(
    expected: PooledDecoderIdentity,
    actual: PooledDecoderIdentity,
) -> list[str]:
    mismatches = []
    if _artifact_path_identity(expected.checkpoint_path) != _artifact_path_identity(
        actual.checkpoint_path
    ):
        mismatches.append(
            f"checkpoint_path: checkpoint={expected.checkpoint_path!r}, "
            f"requested={actual.checkpoint_path!r}"
        )
    for name in ("checkpoint_step", "use_ema", "stage1_profile", "config_sha256"):
        if getattr(expected, name) != getattr(actual, name):
            mismatches.append(
                f"{name}: checkpoint={getattr(expected, name)!r}, "
                f"requested={getattr(actual, name)!r}"
            )
    return mismatches


def _latent_stats_identity_mismatches(
    expected: PooledLatentStatsIdentity,
    actual: PooledLatentStatsIdentity,
) -> list[str]:
    mismatches = []
    if _artifact_path_identity(expected.stats_path) != _artifact_path_identity(
        actual.stats_path
    ):
        mismatches.append(
            f"stats_path: checkpoint={expected.stats_path!r}, "
            f"requested={actual.stats_path!r}"
        )
    for name in ("sha256", "count", "shape"):
        if getattr(expected, name) != getattr(actual, name):
            mismatches.append(
                f"{name}: checkpoint={getattr(expected, name)!r}, "
                f"requested={getattr(actual, name)!r}"
            )
    return mismatches


def load_pooled_latent_stats(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_decoder_identity: PooledDecoderIdentity | None = None,
    require_source_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, PooledLatentStatsIdentity]:
    """Load and validate one normalization-statistics artifact.

    The returned hash covers the complete NPZ bytes.  Source-aware files also
    carry the decoder identity that generated them, preventing a valid-looking
    ``[tokens, channels]`` array from being reused with another tokenizer.
    """

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Latent normalization stats are missing: {path}.")
    with np.load(path, allow_pickle=False) as values:
        missing = {"mean", "var", "count"}.difference(values.files)
        if missing:
            raise ValueError(
                f"Latent stats {path} are missing required entries: {sorted(missing)}."
            )
        mean = np.asarray(values["mean"], dtype=np.float32)
        var = np.asarray(values["var"], dtype=np.float32)
        count = int(_npz_scalar(values, "count"))
        if mean.shape != expected_shape or var.shape != expected_shape:
            raise ValueError(
                f"Latent stats shape mismatch: expected {expected_shape}, "
                f"got mean={mean.shape}, var={var.shape}."
            )
        if count <= 0:
            raise ValueError(f"Latent stats count must be positive, got {count}.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
            raise ValueError("Latent stats contain non-finite mean/variance values.")
        if np.any(var < 0):
            raise ValueError("Latent stats contain negative variance values.")

        if "shape" in values.files:
            stored_shape = tuple(int(v) for v in np.asarray(values["shape"]).tolist())
            if stored_shape != expected_shape:
                raise ValueError(
                    f"Latent-stats shape metadata is {stored_shape}, expected {expected_shape}."
                )
        elif require_source_metadata:
            raise ValueError(
                "RAEv2 source-profile latent stats require explicit shape metadata. "
                "Recompute them with compute_pooled_stats."
            )

        if require_source_metadata:
            sampler_keys = {
                "format_version",
                "source_sample_count",
                "sampler_world_size",
            }
            missing_sampler_keys = sampler_keys.difference(values.files)
            if missing_sampler_keys:
                raise ValueError(
                    "RAEv2 source-profile latent stats lack sampler metadata: "
                    f"{sorted(missing_sampler_keys)}. Recompute them with "
                    "compute_pooled_stats."
                )
            format_version = int(_npz_scalar(values, "format_version"))
            source_sample_count = int(_npz_scalar(values, "source_sample_count"))
            sampler_world_size = int(_npz_scalar(values, "sampler_world_size"))
            if format_version != 1:
                raise ValueError(
                    f"Unsupported pooled latent-stats format version {format_version}."
                )
            if source_sample_count <= 0 or sampler_world_size != 8:
                raise ValueError(
                    "RAEv2 stats require a positive source_sample_count and the "
                    f"released sampler world size 8; got {source_sample_count}, "
                    f"{sampler_world_size}."
                )
            expected_count = (
                (source_sample_count + sampler_world_size - 1)
                // sampler_world_size
                * sampler_world_size
            )
            if count != expected_count:
                raise ValueError(
                    f"RAEv2 stats count {count} does not match the padded source "
                    f"population {expected_count}."
                )

        source_keys = {
            "decoder_path",
            "decoder_step",
            "decoder_use_ema",
            "decoder_stage1_profile",
            "decoder_config_sha256",
        }
        available_source_keys = source_keys.intersection(values.files)
        if available_source_keys and available_source_keys != source_keys:
            raise ValueError(
                "Latent stats contain an incomplete decoder identity: "
                f"missing {sorted(source_keys.difference(values.files))}."
            )
        if source_keys.issubset(values.files):
            stats_decoder_identity = PooledDecoderIdentity(
                checkpoint_path=str(_npz_scalar(values, "decoder_path")),
                checkpoint_step=int(_npz_scalar(values, "decoder_step")),
                use_ema=bool(_npz_scalar(values, "decoder_use_ema")),
                stage1_profile=str(_npz_scalar(values, "decoder_stage1_profile")),
                config_sha256=str(_npz_scalar(values, "decoder_config_sha256")),
            )
            if expected_decoder_identity is not None:
                mismatches = _decoder_identity_mismatches(
                    stats_decoder_identity,
                    expected_decoder_identity,
                )
                if mismatches:
                    raise ValueError(
                        "Latent stats were computed from a different pooled decoder: "
                        + "; ".join(mismatches)
                    )
        elif require_source_metadata:
            raise ValueError(
                "RAEv2 source-profile latent stats do not identify their pooled decoder. "
                "Recompute them with compute_pooled_stats."
            )

    identity = PooledLatentStatsIdentity(
        stats_path=_canonical_artifact_path(path),
        sha256=_file_sha256(path),
        count=count,
        shape=tuple(mean.shape),
    )
    return mean, var, identity


def bind_pooled_generator_artifacts(
    cfg: PooledGeneratorConfig,
    restored: RestoredPooledDecoderComponents,
    stats_identity: PooledLatentStatsIdentity | None,
) -> PooledGeneratorConfig:
    """Attach immutable stage-one/statistics identities to a new run config."""

    if restored.identity.stage1_profile != "raev2_github":
        raise ValueError(
            "Exact raev2_ddt runs require a stage1_profile='raev2_github' "
            f"pooled decoder, got {restored.identity.stage1_profile!r}."
        )
    return replace(
        cfg,
        pooled_decoder_identity=restored.identity,
        latent_stats_identity=stats_identity,
    )


def validate_pooled_generator_artifacts(
    cfg: PooledGeneratorConfig,
    restored: RestoredPooledDecoderComponents,
    stats_identity: PooledLatentStatsIdentity | None,
) -> None:
    """Reject artifact drift when resuming or evaluating a stage-two run."""

    if restored.identity.stage1_profile != "raev2_github":
        raise ValueError(
            "Exact raev2_ddt checkpoints require a stage1_profile='raev2_github' "
            f"pooled decoder, got {restored.identity.stage1_profile!r}."
        )

    if cfg.pooled_decoder_identity is None:
        raise ValueError(
            "RAEv2 DDT checkpoint lacks its pooled-decoder identity; refusing an "
            "unverifiable resume/evaluation."
        )
    else:
        mismatches = _decoder_identity_mismatches(
            cfg.pooled_decoder_identity,
            restored.identity,
        )
        if mismatches:
            raise ValueError(
                "Pooled decoder does not match the generator checkpoint: "
                + "; ".join(mismatches)
            )

    if stats_identity is None:
        raise ValueError("Latent normalization is enabled but no stats identity was loaded.")
    if cfg.latent_stats_identity is None:
        raise ValueError(
            "RAEv2 DDT checkpoint lacks its latent-stats identity; refusing an "
            "unverifiable resume/evaluation."
        )
    stats_mismatches = _latent_stats_identity_mismatches(
        cfg.latent_stats_identity,
        stats_identity,
    )
    if stats_mismatches:
        raise ValueError(
            "Latent stats do not match the generator checkpoint: "
            + "; ".join(stats_mismatches)
        )


def model_policy(cfg: PooledDecoderConfig) -> jmp.Policy:
    compute_dtype = jnp.float32 if cfg.compute_dtype == "float32" else jnp.bfloat16
    return jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=compute_dtype,
        output_dtype=jnp.float32,
    )


def encoder_policy_for_restore(
    mp: jmp.Policy,
    cfg: PooledDecoderConfig,
    *,
    source_encoder_fp32: bool,
) -> jmp.Policy:
    """Decouple exact stage-two RAE encoding from DDT bf16 autocast."""

    if source_encoder_fp32 and getattr(cfg, "stage1_profile", "legacy") == "raev2_github":
        return jmp.Policy(
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            output_dtype=jnp.float32,
        )
    return mp


def _checkpoint_path(path: Path | str) -> Path | str:
    path_str = str(path)
    if path_str.startswith("gs://"):
        return path_str
    local = Path(path_str)
    if not local.exists():
        raise FileNotFoundError(f"Pooled decoder checkpoint path does not exist: {local}")
    return local.absolute()


def _restore_config(
    manager: ocp.CheckpointManager,
    step: int,
) -> tuple[PooledDecoderConfig, str]:
    raw = manager.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))[
        "config"
    ]
    return (
        from_dict(
            PooledDecoderConfig,
            raw,
            config=DaciteConfig(cast=[tuple], strict=False),
        ),
        _canonical_json_sha256(raw),
    )


def _open_pooled_manager(
    path: Path | str,
    *,
    step: int | None,
) -> tuple[ocp.CheckpointManager, int, PooledDecoderConfig, str]:
    checkpoint_path = _checkpoint_path(path)
    manager = ocp.CheckpointManager(
        checkpoint_path,
        item_names=ITEM_NAMES,
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    restore_step = step
    if restore_step is None:
        restore_step = manager.best_step()
    if restore_step is None:
        restore_step = manager.latest_step()
    if restore_step is None:
        raise ValueError(f"No pooled decoder checkpoint found at {path}.")

    cfg, config_sha256 = _restore_config(manager, restore_step)
    if cfg.downsample_mode != "conv":
        return manager, restore_step, cfg, config_sha256

    manager.close()
    manager = ocp.CheckpointManager(
        checkpoint_path,
        item_names=POOLED_ITEM_NAMES,
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    return manager, restore_step, cfg, config_sha256


def restore_pooled_decoder_components(
    path: Path | str,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy | None = None,
    seed: int = 0,
    step: int | None = None,
    use_ema: bool = True,
    restore_decoder: bool = True,
    implementation: str = "xla",
    source_encoder_fp32: bool = False,
    dinov3_checkpoint_path: Path | str | None = None,
) -> RestoredPooledDecoderComponents:
    """Restore DINO, tokenizer, and optionally the image decoder."""

    manager, restore_step, raw_cfg, config_sha256 = _open_pooled_manager(
        path,
        step=step,
    )
    mp = mp or model_policy(raw_cfg)
    # In the released stage-two engine RAE.encode runs before entering the DDT
    # bf16 autocast context. Keep DINO/tokenizer fp32 only for that source path.
    encoder_mp = encoder_policy_for_restore(
        mp,
        raw_cfg,
        source_encoder_fp32=source_encoder_fp32,
    )

    if getattr(raw_cfg, "stage1_profile", "legacy") == "raev2_github":
        dino = load_backbone(
            raw_cfg.dino_name,
            resolution=raw_cfg.backbone_resolution,
            checkpoint_path=dinov3_checkpoint_path,
            persisted_checkpoint_path=getattr(raw_cfg, "dino_checkpoint_path", None),
            dtype=encoder_mp.compute_dtype,
            param_dtype=encoder_mp.param_dtype,
        )
    else:
        if dinov3_checkpoint_path is not None:
            raise ValueError(
                "--dinov3-checkpoint-path is only valid for a DINOv3 pooled decoder."
            )
        dino = load_backbone(
            raw_cfg.dino_name,
            resolution=raw_cfg.backbone_resolution,
            dtype=encoder_mp.param_dtype,
        )
    dino.eval()

    hidden_size = int(dino.config.hidden_size)
    grid_side = raw_cfg.backbone_resolution // dino.patch_size
    grid_hw = (grid_side, grid_side)
    latent_grid = pooled_grid_shape(grid_hw, raw_cfg.pool_window)
    latent_spec = PooledLatentSpec(latent_grid, hidden_size)

    decoder_input_grid = grid_hw if raw_cfg.repeat_pool else latent_grid
    num_tokens = decoder_input_grid[0] * decoder_input_grid[1]
    if getattr(raw_cfg, "stage1_profile", "legacy") == "raev2_github":
        # The released decoder always predicts the full 16x16 RGB patch grid;
        # compressed latent grids are resized inside GeneralDecoder.
        decoder_num_registers = 0
        decoder_vit = replace(
            raw_cfg.vit,
            patch=None,
            num_patches=256,
            input_dim=hidden_size,
            num_registers=0,
            latent_grid_hw=decoder_input_grid,
            latent_upsample="bilinear",
        )
    else:
        decoder_num_registers = 0 if raw_cfg.repeat_pool else 256
        decoder_vit = replace(
            raw_cfg.vit,
            patch=None,
            num_patches=num_tokens,
            input_dim=hidden_size,
            num_registers=decoder_num_registers,
        )
    cfg = replace(
        raw_cfg,
        vit=decoder_vit,
        num_prefix_tokens=dino.num_prefix_tokens,
    )
    num_output_tokens = (
        256
        if getattr(raw_cfg, "stage1_profile", "legacy") == "raev2_github"
        else decoder_num_registers if decoder_num_registers > 0 else num_tokens
    )

    decoder = None
    if restore_decoder:
        decoder_name = "decoder_ema" if use_ema else "decoder"
        decoder = RAEDecoder.restore(
            manager,
            restore_step,
            decoder_name,
            mesh,
            cfg=decoder_vit,
            patch_size=cfg.patch_size,
            num_channels=cfg.out_channels,
            mp=mp,
        )
        decoder.eval()
        set_attn_implementation(decoder, implementation)

    if cfg.downsample_mode == "conv":
        tokenizer_name = "tokenizer_ema" if use_ema else "tokenizer"
        tokenizer = PatchDownsampler.restore(
            manager,
            restore_step,
            tokenizer_name,
            mesh,
            cfg.downsample_mode,
            hidden_size,
            cfg.pool_window,
            encoder_mp,
        )
    else:
        tokenizer = PatchDownsampler(
            cfg.downsample_mode,
            hidden_size,
            cfg.pool_window,
            encoder_mp,
            rngs=nnx.Rngs(seed),
        )

    layer_indices = resolve_representation_layers(
        cfg.representation,
        dino.config.num_hidden_layers,
    )
    manager.close()
    return RestoredPooledDecoderComponents(
        dino=dino,
        decoder=decoder,
        tokenizer=tokenizer,
        cfg=cfg,
        data_cfg=cfg.data,
        aug_cfg=cfg.aug,
        step=restore_step,
        grid_hw=grid_hw,
        latent_spec=latent_spec,
        layer_indices=layer_indices,
        num_output_tokens=num_output_tokens,
        identity=PooledDecoderIdentity(
            checkpoint_path=_canonical_artifact_path(path),
            checkpoint_step=restore_step,
            use_ema=use_ema,
            stage1_profile=getattr(raw_cfg, "stage1_profile", "legacy"),
            config_sha256=config_sha256,
        ),
    )


def configure_for_pooled_latents(
    cfg: PooledGeneratorConfig,
    latent_spec: PooledLatentSpec,
) -> PooledGeneratorConfig:
    return replace(
        cfg,
        raev2_ddt=replace(
            cfg.raev2_ddt,
            input_grid=latent_spec.grid,
            in_channels=latent_spec.feat,
        ),
    )


def make_pooled_generator(
    cfg: PooledGeneratorConfig,
    latent_spec: PooledLatentSpec,
    mp: jmp.Policy,
    rngs: nnx.Rngs,
    *,
    latent_mean: jax.Array | None = None,
    latent_std: jax.Array | None = None,
) -> RAEv2DDT:
    cfg = configure_for_pooled_latents(cfg, latent_spec)
    return RAEv2DDT(
        cfg.raev2_ddt,
        mp,
        latent_mean=latent_mean,
        latent_std=latent_std,
        self_repa_layer=cfg.self_repa_layer if cfg.self_repa else None,
        self_repa_target_grid=(cfg.self_repa_target_grid if cfg.self_repa else None),
        rngs=rngs,
    )


def restore_pooled_generator_item(
    manager: ocp.CheckpointManager,
    step: int,
    item: str,
    *,
    cfg: PooledGeneratorConfig,
    latent_spec: PooledLatentSpec,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    latent_mean: jax.Array | None = None,
    latent_std: jax.Array | None = None,
) -> RAEv2DDT:
    cfg = configure_for_pooled_latents(cfg, latent_spec)
    return RAEv2DDT.restore(
        manager,
        step,
        item,
        mesh,
        cfg.raev2_ddt,
        mp,
        latent_mean=latent_mean,
        latent_std=latent_std,
        self_repa_layer=cfg.self_repa_layer if cfg.self_repa else None,
        self_repa_target_grid=(cfg.self_repa_target_grid if cfg.self_repa else None),
    )


def restore_pooled_generator(
    path: Path | str,
    *,
    latent_spec: PooledLatentSpec,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    use_ema: bool = True,
    step: int | None = None,
    latent_mean: jax.Array | None = None,
    latent_std: jax.Array | None = None,
) -> tuple[RAEv2DDT, PooledGeneratorConfig, int]:
    item_names = ["model", "model_ema", "optim", "loader", "config"]
    manager = ocp.CheckpointManager(
        _checkpoint_path(path),
        item_names=item_names,
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    restore_step = manager.latest_step() if step is None else step
    if restore_step is None:
        raise ValueError(f"No pooled generator checkpoint found at {path}.")
    available_steps = manager.all_steps()
    if restore_step not in available_steps:
        manager.close()
        raise ValueError(
            f"Pooled generator step {restore_step} is unavailable; "
            f"choices: {available_steps}."
        )

    cfg_raw = manager.restore(
        restore_step,
        args=ocp.args.Composite(config=ocp.args.JsonRestore()),
    )["config"]
    cfg = from_dict(
        PooledGeneratorConfig,
        cfg_raw,
        config=DaciteConfig(cast=[tuple], strict=False),
    )
    cfg = configure_for_pooled_latents(cfg, latent_spec)
    item = "model_ema" if use_ema else "model"
    try:
        model = restore_pooled_generator_item(
            manager,
            restore_step,
            item,
            cfg=cfg,
            latent_spec=latent_spec,
            mesh=mesh,
            mp=mp,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
    except KeyError:
        # Only fall back when the EMA item itself is absent. Shape/state-tree
        # mismatches must surface instead of silently switching to raw weights.
        if not use_ema:
            manager.close()
            raise
        manager.close()
        raise ValueError(
            "RAEv2 source evaluation requested model_ema, but this checkpoint "
            "does not contain EMA weights. Refusing to fall back to model."
        ) from None
    manager.close()
    model.eval()
    return model, cfg, restore_step


def encode_pooled_decoder_latents_impl(
    images: jax.Array,
    dino,
    tokenizer: PatchDownsampler,
    *,
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
) -> jax.Array:
    """Return the compressed pre-repeat latent tokens used by pooled decoders."""

    pooled, _ = encode_pooled_decoder_targets_impl(
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
    return pooled


def encode_pooled_decoder_targets_impl(
    images: jax.Array,
    dino,
    tokenizer: PatchDownsampler,
    *,
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
) -> tuple[jax.Array, jax.Array]:
    """Return pooled latents and the exact frozen pre-pooling token field.

    The dense target is reconstructed before calling the tokenizer. In the
    post-pool-normalized ``add_final_mean`` path this means adding the held-out
    final-layer spatial mean back to every full-resolution local token. It is
    never synthesized by repeating pooled tokens and never receives pooled
    latent dataset normalization statistics.
    """

    patches, final_mean = extract_pool_source_tokens(
        images,
        dino,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        split_final_mean=post_pool_norm,
        representation_eps=representation_eps,
    )
    dense_target = patches if final_mean is None else patches + final_mean
    pooled = finish_pooled_tokens(
        jax.random.PRNGKey(0),
        patches,
        final_mean,
        tokenizer,
        post_pool_norm=post_pool_norm,
        representation_eps=representation_eps,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
        repeat_pool=False,
        noise_tau=0.0,
    )
    return (
        jax.lax.stop_gradient(pooled),
        jax.lax.stop_gradient(dense_target),
    )


@nnx.jit(
    static_argnames=(
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
)
def encode_pooled_decoder_latents(
    images: jax.Array,
    dino,
    tokenizer: PatchDownsampler,
    *,
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
) -> jax.Array:
    return encode_pooled_decoder_latents_impl(
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


@nnx.jit(
    static_argnames=(
        "grid_hw",
        "pool_hw",
        "repeat_pool",
        "num_output_tokens",
    )
)
def decode_pooled_decoder_latents(
    decoder: RAEDecoder,
    latents: jax.Array,
    *,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
    repeat_pool: bool,
    num_output_tokens: int,
) -> jax.Array:
    if repeat_pool:
        latents = repeat_pooled_tokens(
            latents,
            pooled_grid_hw=pooled_grid_shape(grid_hw, pool_hw),
            pool_hw=pool_hw,
        )
    tokens = decoder(latents, deterministic=True)[:, :num_output_tokens]
    images = decoder.unpatchify(tokens, denorm_output=True)
    return jnp.clip(images, 0.0, 1.0)


__all__ = [
    "PooledDecoderIdentity",
    "PooledGeneratorConfig",
    "PooledLatentSpec",
    "PooledLatentStatsIdentity",
    "RestoredPooledDecoderComponents",
    "bind_pooled_generator_artifacts",
    "configure_for_pooled_latents",
    "decode_pooled_decoder_latents",
    "encode_pooled_decoder_latents",
    "encode_pooled_decoder_latents_impl",
    "encode_pooled_decoder_targets_impl",
    "encoder_policy_for_restore",
    "load_pooled_latent_stats",
    "make_pooled_generator",
    "model_policy",
    "restore_pooled_decoder_components",
    "restore_pooled_generator",
    "restore_pooled_generator_item",
    "validate_pooled_generator_artifacts",
]
