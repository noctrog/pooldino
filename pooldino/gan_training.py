"""Memory-efficient GAN training primitives for RAE-style pixel decoders.

The original implementation built one VJP over both the image decoder and the
DINO discriminator, then invoked its pullback twice to obtain reconstruction
and adversarial gradients.  At the GAN start epoch this retained two complete
decoder/discriminator backward graphs and also differentiated every
parameter of the discriminator, including its frozen DINO backbone.

This module preserves the adaptive GAN weight used by RAE/RAEv2 while reducing
peak memory:

1. Run the decoder transformer once without differentiation and retain only
   its final hidden tokens.
2. Compute the two adaptive-weight gradient norms with respect to the final
   pixel-projection kernel only.  These are separate compiled executions, so
   LPIPS and discriminator activations are not live simultaneously.
3. Run one ordinary decoder backward pass for the weighted total loss.  The
   discriminator is an input to the loss but is not a differentiation target.

The extra decoder forward trades compute for a substantially smaller peak
activation footprint.  It also mirrors RAEv2's definition of the adaptive
weight: the ratio of reconstruction and GAN gradient norms at the final
pixel-prediction layer.
"""

from __future__ import annotations

from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange
from lpips_nnx import LPIPS

from pooldino.data.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pooldino.diffaug import DiffAug
from pooldino.models.decoder import RAEDecoder
from pooldino.models.discriminator import DinoDisc


NoiseMode = Literal["posterior", "uniform"]
RAEV2_SOURCE_WORLD_SIZE = 8


def raev2_source_rank_groups(
    values: jax.Array,
    world_size: int = RAEV2_SOURCE_WORLD_SIZE,
) -> jax.Array:
    """Group an aggregate batch in DistributedSampler rank order.

    The source rank ``r`` receives global permutation positions
    ``r, r + world_size, ...``. Thus these groups are strided, not contiguous.
    """
    if values.shape[0] % world_size != 0:
        raise ValueError(
            f"Batch size {values.shape[0]} must be divisible by {world_size}."
        )
    grouped = values.reshape(
        values.shape[0] // world_size,
        world_size,
        *values.shape[1:],
    )
    return jnp.swapaxes(grouped, 0, 1)


def raev2_restore_rank_order(groups: jax.Array) -> jax.Array:
    """Invert :func:`raev2_source_rank_groups`."""
    restored = jnp.swapaxes(groups, 0, 1)
    return restored.reshape(-1, *groups.shape[2:])


def raev2_rank_means(
    values: jax.Array,
    world_size: int = RAEV2_SOURCE_WORLD_SIZE,
) -> jax.Array:
    """Mean all non-rank axes independently for each emulated source rank."""
    groups = raev2_source_rank_groups(values, world_size)
    return jnp.mean(groups, axis=tuple(range(1, groups.ndim)))


def raev2_ddp_weighted_loss(
    reconstruction_by_rank: jax.Array,
    perceptual_by_rank: jax.Array,
    adversarial_by_rank: jax.Array,
    adaptive_weight_by_rank: jax.Array,
    *,
    lpips_weight: float,
    gan_weight: float,
) -> jax.Array:
    """DDP-equivalent average of the eight independently weighted losses."""
    local_losses = (
        reconstruction_by_rank
        + lpips_weight * perceptual_by_rank
        + gan_weight
        * jax.lax.stop_gradient(adaptive_weight_by_rank)
        * adversarial_by_rank
    )
    return jnp.mean(local_losses)


def _raev2_rank_diffaug(
    keys: jax.Array,
    images: jax.Array,
    diffaug: DiffAug,
    world_size: int,
) -> jax.Array:
    groups = raev2_source_rank_groups(images, world_size)
    augmented = jax.lax.map(
        lambda pair: diffaug(pair[0], pair[1]),
        (keys, groups),
    )
    return raev2_restore_rank_order(augmented)


def add_latent_noise(
    key: jax.Array,
    mean: jax.Array,
    logvar: jax.Array,
    *,
    tau: float,
    mode: NoiseMode,
) -> jax.Array:
    """Apply either legacy posterior-scaled or RAEv2 feature noising.

    ``posterior`` reproduces the original PoolDINO decoder behavior.
    ``uniform`` samples one sigma per image from U(0, tau), then adds isotropic
    Gaussian noise, matching RAEv2 stage-one decoder training.
    """
    if tau <= 0:
        return mean
    if mode == "posterior":
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        return mean + tau * noise * jnp.exp(0.5 * logvar)
    if mode == "uniform":
        sigma_key, noise_key = jax.random.split(key)
        sigma = tau * jax.random.uniform(
            sigma_key,
            (mean.shape[0],) + (1,) * (mean.ndim - 1),
            dtype=mean.dtype,
        )
        noise = jax.random.normal(noise_key, mean.shape, dtype=mean.dtype)
        return mean + sigma * noise
    raise ValueError(f"Unsupported latent noise mode: {mode!r}.")


def _image_stats(dtype) -> tuple[jax.Array, jax.Array]:
    mean = rearrange(jnp.asarray(IMAGENET_DEFAULT_MEAN, dtype=dtype), "c -> 1 1 1 c")
    std = rearrange(jnp.asarray(IMAGENET_DEFAULT_STD, dtype=dtype), "c -> 1 1 1 c")
    return mean, std


def _unpatchify(
    tokens: jax.Array,
    *,
    patch_size: int,
    num_channels: int,
) -> jax.Array:
    batch, num_tokens, _ = tokens.shape
    side = int(num_tokens**0.5)
    if side * side != num_tokens:
        raise ValueError(f"Output token count {num_tokens} is not a square.")
    return rearrange(
        tokens,
        "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
        h=side,
        w=side,
        p1=patch_size,
        p2=patch_size,
        c=num_channels,
    )


def _decoder_output_tokens(decoder: RAEDecoder) -> int:
    """Number of decoder tokens that should be projected back to image patches."""
    return decoder.num_reg if decoder.num_reg > 0 else decoder.cfg.num_patches


def _images_from_last_layer(
    hidden: jax.Array,
    kernel: jax.Array,
    bias: jax.Array,
    *,
    use_cls: bool,
    num_registers: int,
    patch_size: int,
    num_channels: int,
    image_dtype,
    pixel_output: str,
    clip_generator_inputs: bool,
) -> tuple[jax.Array, jax.Array]:
    """Project detached decoder features to pixel and [-1, 1] image views."""
    projected = jnp.einsum("btd,do->bto", hidden, kernel) + bias
    if use_cls:
        projected = projected[:, 1:]
    projected = projected[:, :num_registers]
    reconstruction = _unpatchify(
        projected,
        patch_size=patch_size,
        num_channels=num_channels,
    )
    image_mean, image_std = _image_stats(image_dtype)
    if pixel_output == "rgb":
        pixels = reconstruction
    elif pixel_output == "imagenet_normalized":
        pixels = reconstruction * image_std + image_mean
    else:
        raise ValueError(f"Unsupported pixel output: {pixel_output!r}.")
    normalized = 2.0 * pixels - 1.0
    if clip_generator_inputs:
        normalized = jnp.clip(normalized, -1.0, 1.0)
    return pixels, normalized


@nnx.jit
def decoder_features_step(decoder: RAEDecoder, latents: jax.Array) -> jax.Array:
    """Run only the decoder transformer and detach its final hidden tokens."""
    hidden = decoder.forward_features(latents, deterministic=True)
    return jax.lax.stop_gradient(hidden)


@nnx.jit(
    static_argnames=(
        "use_lpips",
        "lpips_weight",
        "use_cls",
        "num_registers",
        "patch_size",
        "num_channels",
        "pixel_output",
        "clip_generator_inputs",
    )
)
def reconstruction_last_layer_grad_norm(
    kernel: jax.Array,
    bias: jax.Array,
    hidden: jax.Array,
    images: jax.Array,
    lpips: LPIPS,
    *,
    use_lpips: bool,
    lpips_weight: float,
    use_cls: bool,
    num_registers: int,
    patch_size: int,
    num_channels: int,
    pixel_output: str,
    clip_generator_inputs: bool,
) -> jax.Array:
    """Norm of reconstruction+LPIPS gradient w.r.t. final projection kernel."""
    image_mean, image_std = _image_stats(images.dtype)
    target_pixels = images * image_std + image_mean
    target_lpips = 2.0 * target_pixels - 1.0
    if clip_generator_inputs:
        target_lpips = jnp.clip(target_lpips, -1.0, 1.0)

    def loss_fn(candidate_kernel: jax.Array) -> jax.Array:
        reconstruction, reconstruction_lpips = _images_from_last_layer(
            hidden,
            candidate_kernel,
            bias,
            use_cls=use_cls,
            num_registers=num_registers,
            patch_size=patch_size,
            num_channels=num_channels,
            image_dtype=images.dtype,
            pixel_output=pixel_output,
            clip_generator_inputs=clip_generator_inputs,
        )
        reconstruction_loss = jnp.mean(jnp.abs(reconstruction - target_pixels))
        if use_lpips:
            perceptual_loss = jnp.mean(lpips(reconstruction_lpips, target_lpips))
        else:
            perceptual_loss = jnp.asarray(0.0, dtype=reconstruction_loss.dtype)
        return reconstruction_loss + lpips_weight * perceptual_loss

    gradient = jax.grad(loss_fn)(kernel)
    gradient = gradient.astype(jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.square(gradient)))


@nnx.jit(
    static_argnames=(
        "use_cls",
        "num_registers",
        "patch_size",
        "num_channels",
        "pixel_output",
        "clip_generator_inputs",
    )
)
def gan_last_layer_grad_norm(
    key: jax.Array,
    kernel: jax.Array,
    bias: jax.Array,
    hidden: jax.Array,
    images: jax.Array,
    discriminator: DinoDisc,
    diffaug: DiffAug,
    *,
    use_cls: bool,
    num_registers: int,
    patch_size: int,
    num_channels: int,
    pixel_output: str,
    clip_generator_inputs: bool,
) -> jax.Array:
    """Norm of generator GAN gradient w.r.t. final projection kernel."""

    def loss_fn(candidate_kernel: jax.Array) -> jax.Array:
        _, reconstruction_normalized = _images_from_last_layer(
            hidden,
            candidate_kernel,
            bias,
            use_cls=use_cls,
            num_registers=num_registers,
            patch_size=patch_size,
            num_channels=num_channels,
            image_dtype=images.dtype,
            pixel_output=pixel_output,
            clip_generator_inputs=clip_generator_inputs,
        )
        fake_input = diffaug(key, reconstruction_normalized)
        return -jnp.mean(discriminator(fake_input))

    gradient = jax.grad(loss_fn)(kernel)
    gradient = gradient.astype(jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.square(gradient)))


def compute_adaptive_weight(
    key: jax.Array,
    decoder: RAEDecoder,
    discriminator: DinoDisc,
    images: jax.Array,
    latents: jax.Array,
    diffaug: DiffAug,
    lpips: LPIPS,
    *,
    use_lpips: bool,
    lpips_weight: float,
    max_d_weight: float,
    source_world_size: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute RAEv2's last-layer adaptive weight with bounded peak memory.

    The compiled executions are intentionally orchestrated in Python. Keeping
    them outside one outer ``jit`` prevents XLA from fusing LPIPS and
    discriminator pullbacks into one large live activation graph. The exact
    source profile dispatches one local-batch pullback per emulated rank.
    """
    hidden = decoder_features_step(decoder, latents)
    kernel = decoder.decoder_proj.kernel.value
    bias = decoder.decoder_proj.bias.value
    num_output_tokens = _decoder_output_tokens(decoder)

    common = dict(
        use_cls=decoder.use_cls,
        num_registers=num_output_tokens,
        patch_size=decoder.patch_size,
        num_channels=decoder.num_channels,
        pixel_output=decoder.cfg.pixel_output,
        clip_generator_inputs=decoder.cfg.clip_generator_inputs,
    )
    if source_world_size is None:
        reconstruction_norm = reconstruction_last_layer_grad_norm(
            kernel,
            bias,
            hidden,
            images,
            lpips,
            use_lpips=use_lpips,
            lpips_weight=lpips_weight,
            **common,
        )
        gan_norm = gan_last_layer_grad_norm(
            key,
            kernel,
            bias,
            hidden,
            images,
            discriminator,
            diffaug,
            **common,
        )
    else:
        hidden_groups = raev2_source_rank_groups(hidden, source_world_size)
        image_groups = raev2_source_rank_groups(images, source_world_size)
        rank_keys = jax.random.split(key, source_world_size)
        # Keep these as sequential compiled local-batch pullbacks. This matches
        # the eight independent source processes without materializing eight
        # ViT-XL/LPIPS/DINO gradient graphs concurrently.
        reconstruction_norm = jnp.stack(
            [
                reconstruction_last_layer_grad_norm(
                    kernel,
                    bias,
                    hidden_groups[rank],
                    image_groups[rank],
                    lpips,
                    use_lpips=use_lpips,
                    lpips_weight=lpips_weight,
                    **common,
                )
                for rank in range(source_world_size)
            ]
        )
        gan_norm = jnp.stack(
            [
                gan_last_layer_grad_norm(
                    rank_keys[rank],
                    kernel,
                    bias,
                    hidden_groups[rank],
                    image_groups[rank],
                    discriminator,
                    diffaug,
                    **common,
                )
                for rank in range(source_world_size)
            ]
        )
    adaptive_weight = jnp.clip(
        reconstruction_norm / (gan_norm + 1e-6),
        0.0,
        max_d_weight,
    )
    adaptive_weight = jax.lax.stop_gradient(adaptive_weight)

    # Drop the large detached hidden tensor before dispatching the full decoder
    # backward.  The update depends on adaptive_weight, so device execution is
    # ordered after both norm computations without a host synchronization.
    del hidden
    return adaptive_weight, reconstruction_norm, gan_norm


@nnx.jit(
    static_argnames=(
        "should_update_ema",
        "ema_momentum",
        "use_gan",
        "use_lpips",
        "lpips_weight",
        "source_world_size",
    ),
    donate_argnames=("decoder", "decoder_ema", "optim_dec"),
)
def generator_update_step(
    key: jax.Array,
    decoder: RAEDecoder,
    decoder_ema: RAEDecoder,
    optim_dec: nnx.Optimizer,
    discriminator: DinoDisc,
    images: jax.Array,
    latents: jax.Array,
    adaptive_weight: jax.Array,
    gan_weight: float,
    diffaug: DiffAug,
    lpips: LPIPS,
    *,
    use_gan: bool,
    use_lpips: bool,
    lpips_weight: float,
    ema_momentum: float,
    should_update_ema: bool,
    source_world_size: int | None = None,
) -> dict[str, jax.Array]:
    """One decoder backward pass for reconstruction, LPIPS, and GAN losses."""
    image_mean, image_std = _image_stats(images.dtype)
    target_pixels = images * image_std + image_mean
    target_lpips = 2.0 * target_pixels - 1.0
    if decoder.cfg.clip_generator_inputs:
        target_lpips = jnp.clip(target_lpips, -1.0, 1.0)

    def loss_fn(
        model: RAEDecoder,
        disc: DinoDisc,
        perceptual_model: LPIPS,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        reconstruction_tokens = model(latents, deterministic=True)
        reconstruction_tokens = reconstruction_tokens[
            :, : _decoder_output_tokens(model)
        ]
        reconstruction = model.unpatchify(reconstruction_tokens, denorm_output=False)
        reconstruction_pixels = model.to_pixels(reconstruction)
        reconstruction_normalized = model.to_discriminator_input(reconstruction)

        if source_world_size is None:
            reconstruction_loss = jnp.mean(
                jnp.abs(reconstruction_pixels - target_pixels)
            )
            if use_lpips:
                perceptual_loss = jnp.mean(
                    perceptual_model(reconstruction_normalized, target_lpips)
                )
            else:
                perceptual_loss = jnp.asarray(0.0, dtype=reconstruction_loss.dtype)

            if use_gan:
                fake_input = diffaug(key, reconstruction_normalized)
                adversarial_loss = -jnp.mean(disc(fake_input))
            else:
                adversarial_loss = jnp.asarray(0.0, dtype=reconstruction_loss.dtype)

            reconstruction_total = reconstruction_loss + lpips_weight * perceptual_loss
            total_loss = (
                reconstruction_total
                + gan_weight
                * jax.lax.stop_gradient(adaptive_weight)
                * adversarial_loss
            )
            logged_adaptive_weight = adaptive_weight
        else:
            reconstruction_by_rank = raev2_rank_means(
                jnp.abs(reconstruction_pixels - target_pixels),
                source_world_size,
            )
            if use_lpips:
                perceptual_by_rank = raev2_rank_means(
                    perceptual_model(reconstruction_normalized, target_lpips),
                    source_world_size,
                )
            else:
                perceptual_by_rank = jnp.zeros_like(reconstruction_by_rank)
            if use_gan:
                rank_keys = jax.random.split(key, source_world_size)
                fake_input = _raev2_rank_diffaug(
                    rank_keys,
                    reconstruction_normalized,
                    diffaug,
                    source_world_size,
                )
                adversarial_by_rank = -raev2_rank_means(
                    disc(fake_input),
                    source_world_size,
                )
            else:
                adversarial_by_rank = jnp.zeros_like(reconstruction_by_rank)
            total_loss = raev2_ddp_weighted_loss(
                reconstruction_by_rank,
                perceptual_by_rank,
                adversarial_by_rank,
                adaptive_weight,
                lpips_weight=lpips_weight,
                gan_weight=gan_weight,
            )
            reconstruction_loss = jnp.mean(reconstruction_by_rank)
            perceptual_loss = jnp.mean(perceptual_by_rank)
            adversarial_loss = jnp.mean(adversarial_by_rank)
            logged_adaptive_weight = jnp.mean(adaptive_weight)
        metrics = {
            "loss": total_loss,
            "recon_loss": reconstruction_loss,
            "lpips_loss": perceptual_loss,
            "gan_loss": adversarial_loss,
            "gan_weight": logged_adaptive_weight,
        }
        return total_loss, metrics

    (_, metrics), gradients = nnx.value_and_grad(
        loss_fn,
        argnums=0,
        has_aux=True,
    )(decoder, discriminator, lpips)
    optim_dec.update(decoder, gradients)

    if should_update_ema:
        new_ema_state = jax.tree.map(
            lambda target, source: target * ema_momentum + source * (1.0 - ema_momentum),
            nnx.state(decoder_ema, nnx.Param),
            nnx.state(decoder, nnx.Param),
        )
        nnx.update(decoder_ema, new_ema_state)
    return metrics


def train_generator_step(
    key: jax.Array,
    decoder: RAEDecoder,
    decoder_ema: RAEDecoder,
    optim_dec: nnx.Optimizer,
    discriminator: DinoDisc,
    imgs: jax.Array,
    latents: jax.Array,
    use_gan: bool,
    gan_weight: float,
    diffaug: DiffAug,
    max_d_weight: float,
    use_lpips: bool,
    lpips: LPIPS,
    lpips_weight: float,
    ema_momentum: float,
    should_update_ema: bool = True,
    reuse_adaptive_key: bool = False,
    source_world_size: int | None = None,
) -> dict[str, jax.Array]:
    """Memory-bounded replacement for the original two-pullback update."""
    if use_gan:
        if reuse_adaptive_key:
            adaptive_key = update_key = key
        else:
            adaptive_key, update_key = jax.random.split(key)
        adaptive_weight, reconstruction_norm, gan_norm = compute_adaptive_weight(
            adaptive_key,
            decoder,
            discriminator,
            imgs,
            latents,
            diffaug,
            lpips,
            use_lpips=use_lpips,
            lpips_weight=lpips_weight,
            max_d_weight=max_d_weight,
            source_world_size=source_world_size,
        )
    else:
        update_key = key
        adaptive_weight = jnp.asarray(0.0, dtype=imgs.dtype)
        reconstruction_norm = jnp.asarray(0.0, dtype=jnp.float32)
        gan_norm = jnp.asarray(0.0, dtype=jnp.float32)

    metrics = generator_update_step(
        update_key,
        decoder,
        decoder_ema,
        optim_dec,
        discriminator,
        imgs,
        latents,
        adaptive_weight,
        gan_weight,
        diffaug,
        lpips,
        use_gan=use_gan,
        use_lpips=use_lpips,
        lpips_weight=lpips_weight,
        ema_momentum=ema_momentum,
        should_update_ema=should_update_ema,
        source_world_size=source_world_size,
    )
    metrics["recon_grad_norm"] = jnp.mean(reconstruction_norm)
    metrics["gan_grad_norm"] = jnp.mean(gan_norm)
    return metrics


def discriminator_loss_fn(
    discriminator: DinoDisc,
    real_input: jax.Array,
    fake_input: jax.Array,
) -> jax.Array:
    if discriminator.official_raev2:
        # The released wrapper classifies fake first, then real. This order is
        # observable because spectral-normalization power vectors update on
        # each discriminator forward.
        logits_fake = discriminator(fake_input)
        logits_real = discriminator(real_input)
    else:
        logits_real = discriminator(real_input)
        logits_fake = discriminator(fake_input)
    loss_real = jnp.mean(jax.nn.relu(1.0 - logits_real))
    loss_fake = jnp.mean(jax.nn.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


@nnx.jit(donate_argnames=("discriminator", "optim_disc"))
def train_discriminator_step(
    key: jax.Array,
    discriminator: DinoDisc,
    optim_disc: nnx.Optimizer,
    decoder: RAEDecoder,
    imgs: jax.Array,
    latents: jax.Array,
    diffaug: DiffAug,
) -> dict[str, jax.Array]:
    """Discriminator update with an explicitly detached decoder reconstruction."""
    image_mean, image_std = _image_stats(imgs.dtype)
    fake_tokens = decoder(latents, deterministic=True)
    fake_tokens = fake_tokens[:, : _decoder_output_tokens(decoder)]
    fake_images = decoder.unpatchify(fake_tokens, denorm_output=False)
    fake_images = decoder.to_discriminator_input(fake_images, clip=True)
    fake_images = jax.lax.stop_gradient(jnp.clip(fake_images, -1.0, 1.0))
    real_images = jnp.clip(2.0 * (imgs * image_std + image_mean) - 1.0, -1.0, 1.0)

    # Match RAEv2/VQGAN training by exposing the discriminator to uint8-like
    # generated samples while preserving continuous real images.
    fake_images = jnp.round((fake_images + 1.0) * 127.5) / 127.5 - 1.0
    fake_key, real_key = jax.random.split(key)
    fake_input = diffaug(fake_key, fake_images)
    real_input = diffaug(real_key, real_images)

    loss, gradients = nnx.value_and_grad(discriminator_loss_fn)(
        discriminator,
        real_input,
        fake_input,
    )
    optim_disc.update(discriminator, gradients)
    return {"loss_disc": loss}


RAEV2_DECODER_CONFIGS: dict[str, dict[str, int | str | bool | float]] = {
    "s": {
        "embed_dim": 384,
        "num_layers": 12,
        "mlp_hidden_dim": 1536,
        "num_heads": 6,
    },
    "b": {
        "embed_dim": 768,
        "num_layers": 12,
        "mlp_hidden_dim": 3072,
        "num_heads": 12,
    },
    "l": {
        "embed_dim": 1024,
        "num_layers": 24,
        "mlp_hidden_dim": 4096,
        "num_heads": 16,
    },
    # Exact public RAEv2 ViT-XL decoder dimensions, not pooldino's ViT-XL.
    "xl": {
        "embed_dim": 1152,
        "num_layers": 28,
        "mlp_hidden_dim": 4096,
        "num_heads": 16,
    },
}


def raev2_decoder_transformer_config(size: str) -> dict[str, int | str | bool | float]:
    """Return the public RAEv2 ViT-MAE decoder recipe for a model size."""
    if size not in RAEV2_DECODER_CONFIGS:
        raise ValueError(f"Unknown RAEv2 decoder size: {size!r}.")
    return {
        **RAEV2_DECODER_CONFIGS[size],
        "mlp_type": "gelu",
        "gelu_approximate": False,
        "norm_type": "ln",
        "qk_norm": False,
        "qkv_bias": True,
        "mlp_bias": True,
        "linear_kernel_init": "trunc_normal",
        "linear_init_std": 0.02,
        "layer_norm_eps": 1e-12,
    }


def raev2_github_decoder_transformer_config(
    size: str = "xl",
) -> dict[str, int | str | bool | float]:
    """Released GitHub decoder config, including PyTorch initialization.

    This is separate from the historical ``-raev2`` approximation so existing
    checkpoints keep their original parameterization and initialization.
    """
    return {
        **raev2_decoder_transformer_config(size),
        "linear_kernel_init": "torch_uniform",
        "linear_bias_init": "torch_uniform",
    }


def set_lpips_compute_dtype(lpips: nnx.Module, dtype: jnp.dtype) -> None:
    """Match PyTorch autocast for the frozen LPIPS convolutional layers.

    ``lpips_nnx`` constructs its convolutions with an implicit fp32 compute
    dtype.  RAEv2 evaluates VGG and the learned 1x1 heads inside the stage-one
    autocast context, so its default recipe executes those convolutions in
    bfloat16 while retaining fp32 weights.
    """
    for _, module in nnx.iter_modules(lpips):
        if isinstance(module, nnx.Conv):
            module.dtype = dtype
