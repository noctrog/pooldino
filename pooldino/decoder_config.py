"""Shared configuration/helpers for the PoolDINO paper implementation."""

from dataclasses import dataclass, field, replace

from typing import Literal

import jax.numpy as jnp

import optax

from pooldino.experiment import ExperimentSpec, flag, option, positional

from pooldino.data import DataConfig

from pooldino.models.transformer import TransformerConfig

from pooldino.models.vit import ViTConfig, VIT_CONFIGS

from pooldino.augmentations.decoder import RAEDecoderAugConfig

from pooldino.diffaug import DiffAugConfig

from pooldino.gan_training import NoiseMode, raev2_decoder_transformer_config, train_discriminator_step, train_generator_step

ITEM_NAMES = [
    "decoder",
    "decoder_ema",
    "discriminator",
    "optim_dec",
    "optim_disc",
    "loader",
    "config",
]


@dataclass
class OptimConfig:
    epochs: int = 16
    batch_size: int = 512

    adam_b1: float = 0.9
    adam_b2: float = 0.95
    lr_start: float = 0.0
    lr_peak: float = 2e-4
    lr_final: float = 2e-5
    warmup_epochs: int = 1
    linear_end_epochs: int = 0
    weight_decay: float = 0.0

    ema_decay: float = 0.9978

    lpips_start: int = 0
    disc_start: int = 6
    disc_gan_start: int = 8
    max_d_weight: float = 1e4

    lpips_weight: float = 1.0
    gan_weight: float = 0.75
    schedule_profile: Literal["legacy", "raev2_github"] = "legacy"
    """Opt-in schedule semantics from the released RAEv2 implementation."""


@dataclass
class GeneratorConfig:
    train: OptimConfig = field(default_factory=lambda: OptimConfig())
    vit: ViTConfig = field(
        default_factory=lambda: ViTConfig(
            patch=None,
            num_patches=32,
            num_registers=256,
            transformer=TransformerConfig(**VIT_CONFIGS["vit-s"]),
        )
    )
    data: DataConfig = field(default_factory=lambda: DataConfig())
    aug: RAEDecoderAugConfig = field(default_factory=lambda: RAEDecoderAugConfig())
    diffaug: DiffAugConfig = field(default_factory=lambda: DiffAugConfig(prob=1.0, cutout=0.0))
    patch_size: int = 16
    out_channels: int = 3
    noise_tau: float = 0.2
    noise_mode: NoiseMode = "posterior"
    """Posterior noise preserves legacy runs; RAEv2 uses per-image uniform noise."""
    compute_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    stage1_profile: Literal["legacy", "raev2_github"] = "legacy"
    """Training semantics; legacy is retained for checkpoint compatibility."""


def build_schedules(cfg: OptimConfig, total_updates: int):
    steps_per_epoch = total_updates // cfg.epochs
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    if cfg.schedule_profile == "raev2_github":
        def github_schedule(step):
            step = jnp.asarray(step, dtype=jnp.float32)
            warmup = cfg.lr_peak * (step + 1.0) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_updates - warmup_steps, 1)
            progress = jnp.clip(progress, 0.0, 1.0)
            cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            decay = cfg.lr_final + (cfg.lr_peak - cfg.lr_final) * cosine
            return jnp.where(step < warmup_steps, warmup, decay)

        # The official discriminator owns the same 16-epoch schedule but only
        # advances it on actual updates (starting at epoch six).
        return github_schedule, github_schedule
    if cfg.schedule_profile != "legacy":
        raise ValueError(f"Unknown decoder schedule profile: {cfg.schedule_profile!r}.")
    lr_sched_gen = optax.schedules.warmup_cosine_decay_schedule(
        init_value=cfg.lr_start,
        peak_value=cfg.lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=total_updates,
        end_value=cfg.lr_final,
    )
    lr_sched_disc = optax.schedules.warmup_cosine_decay_schedule(
        init_value=cfg.lr_start,
        peak_value=cfg.lr_peak,
        warmup_steps=steps_per_epoch,
        decay_steps=(cfg.epochs - cfg.disc_start) * steps_per_epoch,
        end_value=cfg.lr_final,
    )
    return lr_sched_gen, lr_sched_disc


_baseline = GeneratorConfig()


def _swiglu_config(vit_name: str) -> dict:
    """Create parameter-matched SwiGLU config from VIT_CONFIGS.

    SwiGLU has 3 weight matrices (up, gate, down) vs GELU's 2 (up, down).
    To match parameters: new_hidden_dim = (2/3) * original_hidden_dim.
    """
    cfg = VIT_CONFIGS[vit_name].copy()
    cfg["mlp_hidden_dim"] = cfg["embed_dim"] * 8 // 3
    cfg["mlp_type"] = "swiglu"
    return cfg


class DecoderExperiment(ExperimentSpec):
    size: Literal["s", "b", "l", "xl"] = positional(prefix="vit")
    fp32: bool = flag()
    raev2: bool = flag()
    e: int | None = option()
    noise: float | None = option()


_DEFAULT_EPOCHS = 16


def build_config(spec: DecoderExperiment) -> GeneratorConfig:
    """Build a legacy PoolDINO or exact RAEv2-style decoder configuration."""
    epochs = spec.e or _DEFAULT_EPOCHS
    if epochs < 16:
        raise ValueError(f"Minimum epochs is 16, got {epochs}")

    compute_dtype: Literal["bfloat16", "float32"] = "float32" if spec.fp32 else "bfloat16"
    train = replace(_baseline.train, epochs=epochs)
    if spec.raev2:
        transformer = replace(
            _baseline.vit.transformer,
            **raev2_decoder_transformer_config(spec.size),
        )
        noise_tau = 0.8 if spec.noise is None else spec.noise
        noise_mode: NoiseMode = "uniform"
    else:
        transformer = replace(
            _baseline.vit.transformer,
            **_swiglu_config(f"vit-{spec.size}"),
        )
        noise_tau = _baseline.noise_tau if spec.noise is None else spec.noise
        noise_mode = "posterior"

    return replace(
        _baseline,
        train=train,
        vit=replace(_baseline.vit, transformer=transformer),
        noise_tau=noise_tau,
        noise_mode=noise_mode,
        compute_dtype=compute_dtype,
    )


def get_experiment(name: str) -> GeneratorConfig:
    spec = DecoderExperiment.parse_or_raise(name)
    return build_config(spec)

