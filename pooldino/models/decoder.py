import jax
import jax.numpy as jnp
import flax.nnx as nnx
import jmp
from einops import rearrange

from pooldino.data.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pooldino.models.transformer import (
    torch_linear_bias_init,
    torch_linear_kernel_init,
)
from pooldino.models.vit import ViTEncoder, ViTConfig


class RAEDecoder(ViTEncoder):
    def __init__(
        self,
        cfg: ViTConfig,
        patch_size: int,
        num_channels: int,
        mp: jmp.Policy,
        *,
        flex: bool = False,
        rngs: nnx.Rngs,
    ):
        super(RAEDecoder, self).__init__(cfg, mp, rngs=rngs)
        self.patch_size = patch_size
        self.num_channels = num_channels

        if cfg.input_proj_init == "torch_uniform":
            output_kernel_init = torch_linear_kernel_init
            output_bias_init = torch_linear_bias_init(cfg.transformer.residual_dim)
        else:
            output_kernel_init = nnx.initializers.truncated_normal(0.02)
            output_bias_init = nnx.initializers.zeros_init()

        self.decoder_proj = nnx.Linear(
            cfg.transformer.residual_dim,
            patch_size**2 * num_channels,
            use_bias=True,
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
            bias_init=output_bias_init,
            kernel_init=output_kernel_init,
            rngs=rngs,
        )

        self.img_mean = nnx.Variable(rearrange(jnp.asarray(IMAGENET_DEFAULT_MEAN), "c -> 1 1 1 c"))
        self.img_std = nnx.Variable(rearrange(jnp.asarray(IMAGENET_DEFAULT_STD), "c -> 1 1 1 c"))

        input_dim = cfg.input_dim or cfg.transformer.residual_dim
        if flex:
            self.mask_embedding = nnx.Param(
                nnx.initializers.truncated_normal(0.02)(
                    rngs(),
                    (1, 1, input_dim),
                    dtype=mp.param_dtype,
                ),
            )
        else:
            self.mask_embedding = None

    def interpolate_latents(self, x: jax.Array) -> jax.Array:
        """Resize a reduced latent grid like RAEv2's ``GeneralDecoder``.

        The released implementation assumes a square grid.  This port also
        accepts an explicit rectangular ``latent_grid_hw`` so the 2x4/4x2
        extensions use the same bilinear operation without guessing geometry.
        """
        if self.cfg.latent_upsample == "none":
            return x
        if self.cfg.latent_upsample != "bilinear":
            raise ValueError(
                f"Unsupported latent upsample mode: {self.cfg.latent_upsample!r}."
            )
        target_tokens = self.cfg.num_patches
        if target_tokens is None:
            raise ValueError("num_patches is required for latent interpolation.")
        if x.shape[1] == target_tokens:
            return x

        if self.cfg.latent_grid_hw is None:
            source_h = source_w = int(x.shape[1] ** 0.5)
        else:
            source_h, source_w = self.cfg.latent_grid_hw
        if source_h * source_w != x.shape[1]:
            raise ValueError(
                f"latent_grid_hw={(source_h, source_w)} does not match "
                f"{x.shape[1]} input tokens."
            )
        target_h = target_w = int(target_tokens**0.5)
        if target_h * target_w != target_tokens:
            raise ValueError(
                f"Decoder target token count {target_tokens} must form a square grid."
            )
        field = x.reshape(x.shape[0], source_h, source_w, x.shape[-1])
        field = jax.image.resize(
            field,
            (x.shape[0], target_h, target_w, x.shape[-1]),
            method="linear",
            antialias=False,
        )
        return field.reshape(x.shape[0], target_tokens, x.shape[-1])

    def forward_features(
        self,
        x: jax.Array | list[jax.Array],
        *,
        deterministic: bool | None = None,
        mask: jax.Array | None = None,
        num_active: jax.Array | None = None,
    ) -> jax.Array:
        """Return the final transformer tokens before the pixel projection.

        Exposing this boundary lets adaptive GAN weighting differentiate only
        the final projection layer instead of retaining two full decoder
        pullbacks.  The normal ``__call__`` path remains unchanged.
        """
        if num_active is not None and self.mask_embedding is not None:
            num_tokens = x.shape[1]
            inactive = jnp.arange(num_tokens)[None, :] >= num_active[:, None]
            x = jnp.where(inactive[:, :, None], self.mask_embedding.value, x)
        if isinstance(x, list):
            raise TypeError("RAE pixel decoder expects one latent tensor, not a list.")
        x = self.interpolate_latents(x)
        return super(RAEDecoder, self).__call__(
            x,
            deterministic=deterministic,
            mask=mask,
        )

    def project_features(self, features: jax.Array) -> jax.Array:
        """Project transformer features to output-patch pixels."""
        x = self.mp.cast_to_compute(features)
        x = self.decoder_proj(x)
        x = x[:, 1:] if self.use_cls else x
        return self.mp.cast_to_output(x)

    def __call__(
        self,
        x: jax.Array | list[jax.Array],
        *,
        deterministic: bool | None = None,
        mask: jax.Array | None = None,
        num_active: jax.Array | None = None,
    ) -> jax.Array:
        features = self.forward_features(
            x,
            deterministic=deterministic,
            mask=mask,
            num_active=num_active,
        )
        return self.project_features(features)

    # TODO: can I just do this in the __call__?
    def unpatchify(self, x: jax.Array, denorm_output: bool = True) -> jax.Array:
        b, t, _ = x.shape
        h = w = int(t**0.5)
        if h * w != t:
            raise ValueError(f"Token count {t} is not a perfect square.")
        recon = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=h,
            w=w,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.num_channels,
        )

        if denorm_output:
            recon = self.to_pixels(recon)

        return recon

    def to_pixels(self, recon: jax.Array) -> jax.Array:
        """Convert the decoder's configured output parameterization to RGB."""
        if self.cfg.pixel_output == "rgb":
            return recon
        if self.cfg.pixel_output == "imagenet_normalized":
            return recon * self.img_std.value + self.img_mean.value
        raise ValueError(f"Unsupported pixel output: {self.cfg.pixel_output!r}.")

    def to_discriminator_input(
        self,
        recon: jax.Array,
        *,
        clip: bool | None = None,
    ) -> jax.Array:
        """Convert raw decoder output to the discriminator/LPIPS [-1,1] view."""
        normalized = 2.0 * self.to_pixels(recon) - 1.0
        should_clip = self.cfg.clip_generator_inputs if clip is None else clip
        return jnp.clip(normalized, -1.0, 1.0) if should_clip else normalized
