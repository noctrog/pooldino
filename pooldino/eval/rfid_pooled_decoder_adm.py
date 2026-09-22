"""Generate reconstructed images for pooled-decoder rFID evaluation.

Exact ``raev2official`` checkpoints reconstruct their configured ImageNet-256
validation backend and use an independent validation NPZ as the real-image
reference. The default Arrow path matches the GitHub RAEv2 evaluator; the
explicit TFDS suffix uses the same Dhariwal preprocessing as a backend ablation.
Legacy checkpoints retain the historical ADM evaluator as a labeled bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Literal

import flax.nnx as nnx
import jax

jax.config.update("jax_default_matmul_precision", "float32")

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import jmp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image
import tensorflow_datasets as tfds
import tyro
from absl import logging
from dacite import Config as DaciteConfig, from_dict
from tqdm import tqdm

from pooldino.data.data import _RAEv2HfImageDataSource
from pooldino.external.adm import get_imagenet_val
from pooldino.models.transformer import set_attn_implementation
from pooldino.backbone import load_backbone
from pooldino.decoder_config import ITEM_NAMES
from pooldino.train_decoder import (
    POOLED_ITEM_NAMES,
    PatchDownsampler,
    PooledDecoderConfig,
    encode_pooled_tokens,
    pooled_grid_shape,
    resolve_representation_layers,
)
from pooldino.augmentations.decoder import RAEv2GithubDecoderAugmentations
from pooldino.models.decoder import RAEDecoder


@dataclass
class Config:
    checkpoint_path: Path
    """Directory containing a pooled-decoder checkpoint."""

    output_dir: Path | None = None
    """Output directory. Defaults to checkpoint_path/recon_samples_mean."""

    num_samples: int | None = None
    """Number of validation images to reconstruct. Defaults to all available images."""

    batch_size: int = 256
    seed: int = 0
    step: int | None = None
    use_ema: bool = True
    implementation: Literal["cudnn", "xla"] = "xla"
    dinov3_checkpoint_path: Path | None = None
    """Override the official DINOv3-L/16 checkpoint used by raev2official."""
    reference_path: Path | None = None
    """Reference uint8 NPZ. Exact runs default to the released ImageNet-256 file."""

    save_npz: bool = True
    save_png: bool = False
    run_adm_evaluator: bool = False
    """Run the selected rFID evaluator after writing samples.npz."""
    rfid_backend: Literal["auto", "adm", "raev2_torch_fidelity"] = "auto"
    """Auto selects torch-fidelity for exact runs and ADM for legacy bridges."""

    wandb_project: str | None = None
    """W&B project to include in the printed ADM evaluator command."""

    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    """Exact W&B run id. Prefer this for logging to the original training run."""

    wandb_experiment: str | None = None
    """Training experiment name. Used with wandb_commit when run id is omitted."""

    wandb_commit: str | None = None
    """Training git commit. Used with wandb_experiment when run id is omitted."""

    wandb_run_name: str | None = None
    wandb_prefix: str = "eval"
    wandb_step: int | None = None


def _fp_to_uint8(images: np.ndarray, *, truncate: bool = False) -> np.ndarray:
    scaled = np.clip(images, 0.0, 1.0) * 255.0
    if not truncate:
        scaled = scaled + 0.5
    return np.clip(scaled, 0, 255).astype(np.uint8)


def permute_raev2_reconstruction_samples(images: np.ndarray) -> np.ndarray:
    """Apply the pinned gather permutation from GitHub RAEv2 reconstruction."""

    permutation = np.random.default_rng(0).permutation(images.shape[0])
    return images[permutation]


def order_reconstruction_samples(
    images: np.ndarray,
    *,
    stage1_profile: str,
) -> tuple[np.ndarray, str]:
    """Resolve the profile-specific metric order and its provenance label."""

    if stage1_profile == "raev2_github":
        return (
            permute_raev2_reconstruction_samples(images),
            "numpy_default_rng_seed_0",
        )
    return images, "none"


def _resolve_reference_path(cfg: Config, train_cfg: PooledDecoderConfig) -> Path:
    if cfg.reference_path is not None:
        path = cfg.reference_path.expanduser()
    elif (
        train_cfg.stage1_profile == "raev2_github"
        and train_cfg.data.backend == "raev2_hf"
    ):
        if train_cfg.data.data_dir is None:
            raise ValueError(
                "Exact RAEv2 rFID requires DataConfig.data_dir or --reference-path."
            )
        path = (
            Path(train_cfg.data.data_dir).expanduser()
            / "imagenet-256-val.npz"
        )
    elif (
        train_cfg.stage1_profile == "raev2_github"
        and train_cfg.data.backend == "tfds"
    ):
        path = Path(get_imagenet_val(train_cfg.aug.crop_size[0]))
    elif train_cfg.stage1_profile == "raev2_github":
        raise ValueError(
            "Exact RAEv2 rFID does not support data backend "
            f"{train_cfg.data.backend!r}."
        )
    else:
        path = Path(get_imagenet_val(train_cfg.aug.crop_size[0]))
    if not path.is_file():
        raise FileNotFoundError(
            f"rFID reference was not found at {path}. Pass --reference-path or "
            "prepare the validation reference for the selected data backend."
        )
    return path


def _resolve_rfid_backend(
    requested: Literal["auto", "adm", "raev2_torch_fidelity"],
    *,
    stage1_profile: str,
) -> Literal["adm", "raev2_torch_fidelity"]:
    if requested == "auto":
        return (
            "raev2_torch_fidelity"
            if stage1_profile == "raev2_github"
            else "adm"
        )
    return requested


def _quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _model_policy(cfg: PooledDecoderConfig) -> jmp.Policy:
    compute_dtype = jnp.float32 if cfg.compute_dtype == "float32" else jnp.bfloat16
    return jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=compute_dtype,
        output_dtype=jnp.float32,
    )


def _resolve_reconstruction_source(
    train_cfg: PooledDecoderConfig,
    ref_images: np.ndarray,
) -> tuple[object, RAEv2GithubDecoderAugmentations | None, str]:
    """Select model inputs independently from the metric reference population."""

    if train_cfg.stage1_profile != "raev2_github":
        return ref_images, None, "reference_npz"
    if train_cfg.data.backend == "tfds":
        source = tfds.data_source(
            train_cfg.data.dataset,
            split=train_cfg.data.val_name,
        )
        augmentation = RAEv2GithubDecoderAugmentations(train_cfg.aug, train_cfg.data)
        return source, augmentation, "tfds_validation_dhariwal_crop"
    if train_cfg.data.backend != "raev2_hf" or train_cfg.data.data_dir is None:
        raise ValueError(
            "Exact RAEv2 rFID requires either backend='raev2_hf' with data_dir "
            "set or backend='tfds'."
        )
    source = _RAEv2HfImageDataSource(
        train_cfg.data.data_dir,
        train_cfg.data.val_name,
    )
    augmentation = RAEv2GithubDecoderAugmentations(train_cfg.aug, train_cfg.data)
    return source, augmentation, "raev2_hf_validation"


def _read_reconstruction_batch(
    source: object,
    augmentation: RAEv2GithubDecoderAugmentations | None,
    start: int,
    end: int,
) -> tuple[np.ndarray, bool]:
    """Read one contiguous model-input batch and report whether it is normalized."""

    if augmentation is None:
        images = np.asarray(source[start:end])
        return images, False
    images = [augmentation.map(source[index])["image"] for index in range(start, end)]
    if not images:
        raise ValueError("Cannot construct an empty reconstruction batch.")
    return np.stack(images, axis=0), True


def _restore_config(
    manager: ocp.CheckpointManager,
    step: int,
) -> PooledDecoderConfig:
    raw = manager.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"]
    return from_dict(
        PooledDecoderConfig,
        raw,
        config=DaciteConfig(cast=[tuple], strict=False),
    )


def _prepare_models(
    cfg: PooledDecoderConfig,
    manager: ocp.CheckpointManager,
    step: int,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    use_ema: bool,
    implementation: str,
    dinov3_checkpoint_path: Path | None,
) -> tuple[
    PooledDecoderConfig,
    object,
    RAEDecoder,
    PatchDownsampler,
    tuple[int, int],
    tuple[int, ...],
    int,
]:
    if cfg.stage1_profile == "raev2_github":
        dino = load_backbone(
            cfg.dino_name,
            resolution=cfg.backbone_resolution,
            checkpoint_path=dinov3_checkpoint_path,
            persisted_checkpoint_path=cfg.dino_checkpoint_path,
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
        )
    else:
        dino = load_backbone(
            cfg.dino_name,
            resolution=cfg.backbone_resolution,
            dtype=mp.param_dtype,
        )
    dino.eval()

    hidden_size = int(dino.config.hidden_size)
    grid_side = cfg.backbone_resolution // dino.patch_size
    grid_hw = (grid_side, grid_side)
    pooled_grid = pooled_grid_shape(grid_hw, cfg.pool_window)
    decoder_input_grid = grid_hw if cfg.repeat_pool else pooled_grid
    num_tokens = decoder_input_grid[0] * decoder_input_grid[1]
    if cfg.stage1_profile == "raev2_github":
        decoder_num_registers = 0
        decoder_vit = replace(
            cfg.vit,
            patch=None,
            num_patches=256,
            input_dim=hidden_size,
            num_registers=0,
            latent_grid_hw=decoder_input_grid,
            latent_upsample="bilinear",
        )
    else:
        decoder_num_registers = 0 if cfg.repeat_pool else 256
        decoder_vit = replace(
            cfg.vit,
            patch=None,
            num_patches=num_tokens,
            input_dim=hidden_size,
            num_registers=decoder_num_registers,
        )
    cfg = replace(
        cfg,
        vit=decoder_vit,
        num_prefix_tokens=dino.num_prefix_tokens,
    )
    num_output_tokens = (
        256
        if cfg.stage1_profile == "raev2_github"
        else decoder_num_registers if decoder_num_registers > 0 else num_tokens
    )
    output_side = int(num_output_tokens**0.5)
    if output_side * output_side != num_output_tokens:
        raise ValueError(f"Decoder output token count {num_output_tokens} is not square.")

    item_name = "decoder_ema" if use_ema else "decoder"
    decoder = RAEDecoder.restore(
        manager,
        step,
        item_name,
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
            step,
            tokenizer_name,
            mesh,
            cfg.downsample_mode,
            hidden_size,
            cfg.pool_window,
            mp,
        )
    else:
        tokenizer = PatchDownsampler(
            cfg.downsample_mode,
            hidden_size,
            cfg.pool_window,
            mp,
            rngs=nnx.Rngs(0),
        )

    layer_indices = resolve_representation_layers(
        cfg.representation,
        dino.config.num_hidden_layers,
    )
    return cfg, dino, decoder, tokenizer, grid_hw, layer_indices, num_output_tokens


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
        "repeat_pool",
        "num_output_tokens",
        "inputs_normalized",
    )
)
def _reconstruct_batch(
    images: jax.Array,
    key: jax.Array,
    dino,
    decoder: RAEDecoder,
    tokenizer: PatchDownsampler,
    mean: jax.Array,
    std: jax.Array,
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
    repeat_pool: bool,
    num_output_tokens: int,
    inputs_normalized: bool,
) -> jax.Array:
    if inputs_normalized:
        images_norm = images.astype(jnp.float32)
    else:
        images_fp = images.astype(jnp.float32) / 255.0
        images_norm = (images_fp - mean) / std
    pooled = encode_pooled_tokens(
        key,
        images_norm,
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
        repeat_pool=repeat_pool,
        noise_tau=0.0,
    )
    tokens = decoder(pooled, deterministic=True)[:, :num_output_tokens]
    recon = decoder.unpatchify(tokens, denorm_output=True)
    return jnp.clip(recon, 0.0, 1.0)


def _wandb_args(cfg: Config) -> list[str]:
    if cfg.wandb_project is None:
        return []
    args = ["--wandb-project", cfg.wandb_project, "--wandb-prefix", cfg.wandb_prefix]
    if cfg.wandb_entity is not None:
        args.extend(["--wandb-entity", cfg.wandb_entity])
    if cfg.wandb_run_id is not None:
        args.extend(["--wandb-run-id", cfg.wandb_run_id])
    if cfg.wandb_experiment is not None:
        args.extend(["--wandb-experiment", cfg.wandb_experiment])
    if cfg.wandb_commit is not None:
        args.extend(["--wandb-commit", cfg.wandb_commit])
    if cfg.wandb_run_name is not None:
        args.extend(["--wandb-run-name", cfg.wandb_run_name])
    if cfg.wandb_step is not None:
        args.extend(["--wandb-step", str(cfg.wandb_step)])
    return args


def main(cfg: Config) -> None:
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if cfg.run_adm_evaluator and not cfg.save_npz:
        raise ValueError("--run-adm-evaluator requires --save-npz.")
    output_dir = cfg.output_dir or cfg.checkpoint_path / "recon_samples_mean"
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    if cfg.batch_size % mesh.size != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by device count ({mesh.size})."
        )

    manager = ocp.CheckpointManager(
        cfg.checkpoint_path.absolute(),
        item_names=ITEM_NAMES,
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    step = cfg.step
    if step is None:
        step = manager.best_step() or manager.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found in {cfg.checkpoint_path}.")
    logging.info("Restoring pooled decoder checkpoint at step %d", step)

    train_cfg = _restore_config(manager, step)
    if train_cfg.downsample_mode == "conv":
        manager.close()
        manager = ocp.CheckpointManager(
            cfg.checkpoint_path.absolute(),
            item_names=POOLED_ITEM_NAMES,
            options=ocp.CheckpointManagerOptions(read_only=True),
        )
    mp = _model_policy(train_cfg)
    train_cfg, dino, decoder, tokenizer, grid_hw, layer_indices, num_output_tokens = _prepare_models(
        train_cfg,
        manager,
        step,
        mesh=mesh,
        mp=mp,
        use_ema=cfg.use_ema,
        implementation=cfg.implementation,
        dinov3_checkpoint_path=cfg.dinov3_checkpoint_path,
    )

    reference_path = _resolve_reference_path(cfg, train_cfg)
    rfid_backend = _resolve_rfid_backend(
        cfg.rfid_backend,
        stage1_profile=train_cfg.stage1_profile,
    )
    ref_images = np.load(reference_path)["arr_0"]
    if ref_images.dtype != np.uint8 or ref_images.ndim != 4 or ref_images.shape[-1] != 3:
        raise ValueError(
            f"rFID reference must contain uint8 NHWC RGB images, got "
            f"{ref_images.dtype} {ref_images.shape}."
        )
    reconstruction_source, reconstruction_augmentation, reconstruction_source_name = (
        _resolve_reconstruction_source(train_cfg, ref_images)
    )
    total_available = len(reconstruction_source)
    num_samples = cfg.num_samples if cfg.num_samples is not None else total_available
    if not 0 < num_samples <= total_available:
        raise ValueError(
            f"num_samples must lie in [1, {total_available}], got {num_samples}."
        )

    mean = jnp.asarray(train_cfg.data.normalization_mean, dtype=jnp.float32)[None, None, None, :]
    std = jnp.asarray(train_cfg.data.normalization_std, dtype=jnp.float32)[None, None, None, :]
    shard = NamedSharding(mesh, P("data"))
    key = jax.random.PRNGKey(cfg.seed)
    reconstruct = partial(
        _reconstruct_batch,
        dino=dino,
        decoder=decoder,
        tokenizer=tokenizer,
        mean=mean,
        std=std,
        backbone_resolution=train_cfg.backbone_resolution,
        num_prefix_tokens=train_cfg.num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=train_cfg.representation.aggregation,
        normalize_each=train_cfg.representation.normalize_each,
        add_final_mean=train_cfg.representation.add_final_mean,
        post_pool_norm=train_cfg.post_pool_norm,
        representation_eps=train_cfg.representation.eps,
        grid_hw=grid_hw,
        pool_hw=train_cfg.pool_window,
        repeat_pool=train_cfg.repeat_pool,
        num_output_tokens=num_output_tokens,
    )

    all_samples = []
    total_collected = 0
    pbar = tqdm(total=num_samples, desc="Reconstructing")
    for start in range(0, num_samples, cfg.batch_size):
        end = min(start + cfg.batch_size, num_samples)
        images, inputs_normalized = _read_reconstruction_batch(
            reconstruction_source,
            reconstruction_augmentation,
            start,
            end,
        )
        batch_len = images.shape[0]
        pad_size = (-batch_len) % mesh.size
        if pad_size:
            images = np.concatenate([images, np.repeat(images[-1:], pad_size, axis=0)], axis=0)
        images = jax.device_put(jnp.asarray(images), shard)
        key, batch_key = jax.random.split(key)
        recon = np.asarray(
            jax.device_get(
                reconstruct(
                    images,
                    batch_key,
                    inputs_normalized=inputs_normalized,
                )
            )
        )
        if pad_size:
            recon = recon[:-pad_size]
        recon_uint8 = _fp_to_uint8(
            recon,
            # RAEv2 uses ``mul(255).to(uint8)`` (truncation), while the ADM
            # helper historically rounded to nearest.
            truncate=train_cfg.stage1_profile == "raev2_github",
        )
        if cfg.save_npz:
            all_samples.append(recon_uint8)
        if cfg.save_png:
            image_dir = output_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            for idx, image in enumerate(recon_uint8):
                Image.fromarray(image).save(image_dir / f"sample_{total_collected + idx:06d}.png")
        total_collected += batch_len
        pbar.update(batch_len)
    pbar.close()
    manager.close()

    npz_path = output_dir / "samples.npz"
    reconstruction_permutation = (
        "numpy_default_rng_seed_0"
        if train_cfg.stage1_profile == "raev2_github"
        else "none"
    )
    if cfg.save_npz:
        samples = np.concatenate(all_samples, axis=0)
        # Released gather shuffles reconstructions after rank-order
        # concatenation. The independent real reference remains untouched.
        samples, reconstruction_permutation = order_reconstruction_samples(
            samples,
            stage1_profile=train_cfg.stage1_profile,
        )
        np.savez(npz_path, arr_0=samples)
        logging.info("Saved %d reconstructed samples to %s", samples.shape[0], npz_path)

    config_path = output_dir / "reconstruction_config.txt"
    with open(config_path, "w") as file:
        file.write(f"reconstruction_source: {reconstruction_source_name}\n")
        file.write(f"reconstruction_permutation: {reconstruction_permutation}\n")
        file.write("reference_permutation: none\n")
        file.write(f"reference_path: {reference_path}\n")
        file.write(f"rfid_backend: {rfid_backend}\n")
        file.write(f"checkpoint_path: {cfg.checkpoint_path}\n")
        file.write(f"checkpoint_step: {step}\n")
        file.write(f"num_samples: {num_samples}\n")
        file.write(f"total_reconstructed: {total_collected}\n")
        file.write(f"use_ema: {cfg.use_ema}\n")
        file.write(f"seed: {cfg.seed}\n")
        file.write(f"pool_window: {train_cfg.pool_window}\n")
        file.write(f"repeat_pool: {train_cfg.repeat_pool}\n")
        file.write(f"post_pool_norm: {train_cfg.post_pool_norm}\n")
        file.write(f"downsample_mode: {train_cfg.downsample_mode}\n")
        file.write(f"representation: {train_cfg.representation}\n")

    logging.info("")
    metric_reference_path = reference_path
    if num_samples != total_available:
        # Released reconstruction.py slices only the reference population to
        # the reconstruction count. It does not truncate reconstructions when
        # the reference is already shorter.
        metric_reference_path = output_dir / "reference_samples.npz"
        np.savez(metric_reference_path, arr_0=ref_images[:num_samples])

    if rfid_backend == "raev2_torch_fidelity":
        command = [
            sys.executable,
            "-m",
            "pooldino.eval.raev2_rfid",
            "--reference-path",
            str(metric_reference_path),
            "--reconstruction-path",
            str(npz_path),
            "--batch-size",
            str(min(cfg.batch_size, 128)),
            *_wandb_args(cfg),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "pooldino.external.adm.evaluator",
            str(npz_path),
            "--ref-batch",
            str(metric_reference_path),
            *_wandb_args(cfg),
        ]
    if cfg.run_adm_evaluator:
        logging.info("Running %s evaluator:", rfid_backend)
        logging.info("  %s", _quote_command(command))
        subprocess.run(command, check=True)
    else:
        logging.info("To compute rFID and optionally log it to W&B, run:")
        logging.info("  %s", _quote_command(command))


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(tyro.cli(Config))
