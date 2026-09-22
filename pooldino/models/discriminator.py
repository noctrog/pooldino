from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.nnx.nn import dtypes
import chex
import jmp
from einops import rearrange

from pooldino.pretrained.raev2_dino_discriminator import RAEv2DinoS8
from pooldino.utils import Restorable, restore_module_state

# ImageNet normalization constants for DINO discriminator
# Converts input from [-1, 1] to ImageNet normalized space
# Formula: x_norm = x * x_scale + x_shift where x in [-1, 1]
# Derived from: ((x+1)/2 - mean) / std = 0.5*x/std + (0.5-mean)/std
IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])
DINO_X_SCALE = (0.5 / IMAGENET_STD).reshape(1, 1, 1, 3)  # For BHWC format
DINO_X_SHIFT = ((0.5 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, 1, 1, 3)


def _torch_conv_kernel_init(key, shape, dtype=jnp.float32):
    """PyTorch Conv1d default initialization for NNX's KIO layout."""
    if len(shape) != 3:
        raise ValueError(f"Expected a rank-3 Conv1d kernel, got {shape}.")
    fan_in = shape[0] * shape[1]
    bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=jnp.float32))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)


def _torch_conv_bias_init(fan_in: int):
    bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=jnp.float32))

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key,
            shape,
            dtype=dtype,
            minval=-bound,
            maxval=bound,
        )

    return init


def _l2_normalize(x: jax.Array, eps: float) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.sum(jnp.square(x), keepdims=True) + eps)


def _conv_dimension_numbers(input_shape) -> jax.lax.ConvDimensionNumbers:
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def _canonicalize_padding(padding, rank: int):
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, (tuple, list)) and len(padding) == rank:
        out = []
        for pad in padding:
            if isinstance(pad, int):
                out.append((pad, pad))
            elif isinstance(pad, tuple) and len(pad) == 2:
                out.append(pad)
            else:
                break
        if len(out) == rank:
            return out
    raise ValueError(f"Invalid padding format: {padding!r}.")


def _maybe_broadcast_conv_dim(x, rank: int) -> tuple[int, ...]:
    if x is None:
        x = 1
    if isinstance(x, int):
        return (x,) * rank
    return tuple(x)


def _conv_with_kernel(conv: nnx.Conv, inputs: jax.Array, kernel: jax.Array) -> jax.Array:
    """Call an NNX Conv using a supplied kernel without mutating the Param."""
    kernel_size = conv.kernel_size
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
        input_batch_shape = inputs.shape[:num_batch_dimensions]
        inputs = jnp.reshape(inputs, (-1,) + inputs.shape[num_batch_dimensions:])

    strides = _maybe_broadcast_conv_dim(conv.strides, len(kernel_size))
    input_dilation = _maybe_broadcast_conv_dim(conv.input_dilation, len(kernel_size))
    kernel_dilation = _maybe_broadcast_conv_dim(conv.kernel_dilation, len(kernel_size))

    padding_lax = _canonicalize_padding(conv.padding, len(kernel_size))
    if padding_lax in ("CIRCULAR", "REFLECT"):
        kernel_size_dilated = [
            (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation, strict=True)
        ]
        pads = (
            [(0, 0)]
            + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
            + [(0, 0)]
        )
        padding_mode = {"CIRCULAR": "wrap", "REFLECT": "reflect"}[padding_lax]
        inputs = jnp.pad(inputs, pads, mode=padding_mode)
        padding_lax = "VALID"
    elif padding_lax == "CAUSAL":
        if len(kernel_size) != 1:
            raise ValueError("Causal padding is only implemented for 1D convolutions.")
        left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
        inputs = jnp.pad(inputs, [(0, 0), (left_pad, 0), (0, 0)])
        padding_lax = "VALID"

    if conv.mask is not None:
        if conv.mask.shape != conv.kernel_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {conv.mask.shape}, {conv.kernel_shape}"
            )
        kernel = kernel * conv.mask

    bias = conv.bias[...] if conv.bias is not None else None
    inputs, kernel, bias = conv.promote_dtype((inputs, kernel, bias), dtype=conv.dtype)

    conv_kwargs = {}
    if conv.preferred_element_type is not None:
        conv_kwargs["preferred_element_type"] = conv.preferred_element_type

    y = conv.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding_lax,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=_conv_dimension_numbers(inputs.shape),
        feature_group_count=conv.feature_group_count,
        precision=conv.precision,
        **conv_kwargs,
    )

    if conv.use_bias:
        y = y + bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)

    if num_batch_dimensions != 1:
        y = jnp.reshape(y, input_batch_shape + y.shape[1:])
    return y


class EvalSafeSpectralNorm(nnx.SpectralNorm):
    """Spectral norm with an exact, non-mutating RAEv2 path.

    Flax NNX normalizes the wrapped layer's parameter in place.  PyTorch's
    legacy ``torch.nn.utils.spectral_norm`` instead keeps ``weight_orig`` as
    the optimizer parameter and uses an ephemeral ``weight_orig / sigma`` for
    the convolution.  It also stores *both* power vectors: one iteration
    updates them in training, while evaluation performs zero iterations and
    computes sigma from the stored vectors.  ``official_raev2`` implements
    those source semantics; the default retains pooldino-code's historical behavior.
    """

    def __init__(
        self,
        layer_instance: nnx.Module,
        *,
        official_raev2: bool = False,
        n_steps: int = 1,
        epsilon: float = 1e-12,
        dtype=None,
        param_dtype=jnp.float32,
        error_on_non_matrix: bool = False,
        update_stats: bool = True,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            layer_instance,
            n_steps=n_steps,
            epsilon=epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            error_on_non_matrix=error_on_non_matrix,
            update_stats=update_stats,
            rngs=rngs,
        )
        self.official_raev2 = official_raev2
        if not official_raev2:
            return

        # ``SpectralNorm.apply`` initializes independent, unit-normalized u/v
        # vectors.  NNX only creates u and sigma, so replace its unnormalized u
        # and add the missing v buffer for the exact path.
        state = nnx.state(self.layer_instance, nnx.Param)
        for path, param in nnx.to_flat_state(state):
            if param.ndim <= 1 or self.n_steps < 1:
                continue
            if param.ndim > 2:
                if self.error_on_non_matrix:
                    raise ValueError(
                        f"Layer instance parameter is {param.ndim}D but "
                        "error_on_non_matrix is True"
                    )
                # PyTorch Conv1d flattens OIK to O x (I*K). NNX stores KIO,
                # so put input channels before spatial axes before flattening.
                order = (param.ndim - 2, *range(param.ndim - 2), param.ndim - 1)
                param = jnp.transpose(param, order)
                param = jnp.reshape(param, (-1, param.shape[-1]))
            path_u = path + ("u",)
            path_v = path + ("v",)
            u = _l2_normalize(self.batch_stats[path_u][...], eps=self.epsilon)
            v = _l2_normalize(
                jax.random.normal(
                    rngs.params(),
                    (1, param.shape[0]),
                    dtype=self.param_dtype,
                ),
                eps=self.epsilon,
            )
            self.batch_stats[path_u][...] = u
            self.batch_stats[path_v] = nnx.BatchStat(v)

    def __call__(self, x: jax.Array, update_stats: bool | None = None) -> jax.Array:
        update_stats = not self.use_running_average if update_stats is None else update_stats
        if self.official_raev2:
            if not isinstance(self.layer_instance, nnx.Conv):
                raise TypeError("RAEv2 spectral norm currently supports nnx.Conv only.")
            kernel = self._raev2_normalized_param(
                ("kernel",),
                self.layer_instance.kernel[...],
                update_stats=update_stats,
            )
            return _conv_with_kernel(self.layer_instance, x, kernel)
        if update_stats:
            return super().__call__(x, update_stats=True)

        if not isinstance(self.layer_instance, nnx.Conv):
            raise TypeError("EvalSafeSpectralNorm currently supports nnx.Conv layers only.")
        kernel = self._normalized_param(("kernel",), self.layer_instance.kernel[...])
        return _conv_with_kernel(self.layer_instance, x, kernel)

    def _normalized_param(self, path: tuple[str, ...], param: jax.Array) -> jax.Array:
        param_shape = param.shape
        if param.ndim <= 1 or self.n_steps < 1:
            return param
        if param.ndim > 2:
            if self.error_on_non_matrix:
                raise ValueError(
                    f"Layer instance parameter is {param.ndim}D but error_on_non_matrix is True"
                )
            param = jnp.reshape(param, (-1, param.shape[-1]))

        u = self.batch_stats[path + ("u",)][...]
        for _ in range(self.n_steps):
            v = _l2_normalize(jnp.matmul(u, param.T), eps=self.epsilon)
            u = _l2_normalize(jnp.matmul(v, param), eps=self.epsilon)

        u = jax.lax.stop_gradient(u)
        v = jax.lax.stop_gradient(v)
        sigma = jnp.matmul(jnp.matmul(v, param), u.T)[0, 0]
        param = param / jnp.where(sigma != 0, sigma, 1)
        param = jnp.reshape(param, param_shape)

        dtype = dtypes.canonicalize_dtype(param, u, v, sigma, dtype=self.dtype)
        return jnp.asarray(param, dtype)

    def _raev2_normalized_param(
        self,
        path: tuple[str, ...],
        param: jax.Array,
        *,
        update_stats: bool,
    ) -> jax.Array:
        """Normalize an ephemeral view of raw W with PyTorch hook semantics."""
        param_shape = param.shape
        if param.ndim <= 1 or self.n_steps < 1:
            return param
        restore_order = None
        if param.ndim > 2:
            if self.error_on_non_matrix:
                raise ValueError(
                    f"Layer instance parameter is {param.ndim}D but "
                    "error_on_non_matrix is True"
                )
            order = (param.ndim - 2, *range(param.ndim - 2), param.ndim - 1)
            restore_order = tuple(sorted(range(param.ndim), key=order.__getitem__))
            reordered_shape = tuple(param.shape[index] for index in order)
            param = jnp.transpose(param, order)
            param = jnp.reshape(param, (-1, param.shape[-1]))

        path_u = path + ("u",)
        path_v = path + ("v",)
        path_sigma = path + ("sigma",)
        u = self.batch_stats[path_u][...]
        v = self.batch_stats[path_v][...]

        # PyTorch performs power iteration under no_grad only in train mode.
        # Evaluation must use stored u/v directly (zero power iterations).
        if update_stats:
            for _ in range(self.n_steps):
                v = _l2_normalize(jnp.matmul(u, param.T), eps=self.epsilon)
                u = _l2_normalize(jnp.matmul(v, param), eps=self.epsilon)
            u = jax.lax.stop_gradient(u)
            v = jax.lax.stop_gradient(v)
            self.batch_stats[path_u][...] = u
            self.batch_stats[path_v][...] = v
        else:
            u = jax.lax.stop_gradient(u)
            v = jax.lax.stop_gradient(v)

        sigma = jnp.matmul(jnp.matmul(v, param), u.T)[0, 0]
        if update_stats:
            self.batch_stats[path_sigma][...] = jax.lax.stop_gradient(sigma)
        normalized = param / jnp.where(sigma != 0, sigma, 1)
        if restore_order is None:
            normalized = jnp.reshape(normalized, param_shape)
        else:
            normalized = jnp.reshape(normalized, reordered_shape)
            normalized = jnp.transpose(normalized, restore_order)
        dtype = dtypes.canonicalize_dtype(
            normalized,
            u,
            v,
            sigma,
            dtype=self.dtype,
        )
        return jnp.asarray(normalized, dtype)


class BatchNormLocal(nnx.Module):
    def __init__(
        self,
        num_features: int,
        mp: jmp.Policy,  # Assuming jmp is imported
        affine: bool = True,
        virtual_bs: int = 1,
        eps: float = 1e-6,
        cast_output_to_compute: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine
        self.mp = mp
        self.cast_output_to_compute = cast_output_to_compute

        if self.affine:
            self.weight = nnx.Param(jnp.ones((num_features,), dtype=mp.param_dtype))
            self.bias = nnx.Param(jnp.zeros((num_features,), dtype=mp.param_dtype))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 3)  # Input: (Batch, Time, Features)
        x = jnp.astype(x, jnp.float32)
        x = rearrange(x, "(g v) t d -> g v t d", v=self.virtual_bs)

        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.var(x, axis=(1, 2), keepdims=True)
        scale = jax.lax.rsqrt(var + self.eps)
        x = (x - mean) * scale

        if self.affine:
            w = self.weight.value[None, None, None, :]
            b = self.bias.value[None, None, None, :]
            x = x * w + b

        x = rearrange(x, "g v t d -> (g v) t d")
        # RAEv2's PyTorch BatchNormLocal starts with ``x.float()`` and returns
        # float32 even under autocast.  Legacy pooldino-code heads historically cast
        # back to the mixed-precision compute dtype.
        return self.mp.cast_to_compute(x) if self.cast_output_to_compute else x


class ResidualBlock(nnx.Module):
    def __init__(self, fn: Callable):
        self.fn = fn
        self.ratio = 2**-0.5

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        return (self.fn(x, **kwargs) + x) * self.ratio


class DinoDiscBlock(nnx.Module):
    def __init__(
        self,
        mp: jmp.Policy,
        channels: int,
        kernel_size: int,
        norm_type: str,
        *,
        official_raev2: bool = False,
        rngs: nnx.Rngs,
    ):
        # This value is autamatically set with nnx.Module.{train(), eval()}. But nnx.SpectralNorm
        # takes an `update_stats` argument for training/eval. We store this dummy variable here
        # to track the training/eval state instead. By default, it's on training state
        self.use_running_average = False

        match norm_type:
            case "bn":
                self.norm = BatchNormLocal(
                    channels,
                    mp,
                    cast_output_to_compute=not official_raev2,
                    rngs=rngs,
                )
            case "gn":
                self.norm = nnx.GroupNorm(channels, num_groups=32, rngs=rngs)
            case _:
                raise ValueError("Invalid norm type")

        conv_kwargs = {}
        if official_raev2:
            conv_kwargs = {
                "kernel_init": _torch_conv_kernel_init,
                "bias_init": _torch_conv_bias_init(channels * kernel_size),
                "dtype": mp.compute_dtype,
                "param_dtype": mp.param_dtype,
            }
        self.conv = EvalSafeSpectralNorm(
            nnx.Conv(
                channels,
                channels,
                kernel_size=(kernel_size,),
                padding="CIRCULAR",
                rngs=rngs,
                **conv_kwargs,
            ),
            official_raev2=official_raev2,
            rngs=rngs,
        )
        self.relu = partial(nnx.leaky_relu, negative_slope=0.2)

    def __call__(self, x: jax.Array):
        is_training = not self.use_running_average
        x = self.conv(x, update_stats=is_training)
        x = self.norm(x)
        return self.relu(x)


# def make_block(
#     mp: jmp.Policy,
#     channels: int,
#     kernel_size: int,
#     norm_type: str,
#     *,
#     rngs: nnx.Rngs,
# ) -> nnx.Module:
#     match norm_type:
#         case "bn":
#             norm = BatchNormLocal(channels, mp, rngs=rngs)
#         case "gn":
#             norm = nnx.GroupNorm(channels, num_groups=32, rngs=rngs)
#         case _:
#             raise ValueError("Invalid norm type")

#     # TODO: this applies the spectral norm also to the bias, and the ref. impl. does not
#     conv = nnx.SpectralNorm(
#         nnx.Conv(channels, channels, kernel_size=(kernel_size,), padding="CIRCULAR", rngs=rngs),
#         rngs=rngs,
#     )

#     return nnx.Sequential(conv, norm, partial(nnx.leaky_relu, negative_slope=0.2))


class DinoDisc(nnx.Module, Restorable):
    @classmethod
    def restore(cls, mngr, step: int, name: str, mesh=None, *args, **kwargs):
        """Restore without ``nnx.eval_shape`` to keep the Linen DINO params concrete."""
        model = cls(*args, **kwargs, rngs=nnx.Rngs(0))
        restore_module_state(mngr, step, model, mesh, name)
        return model

    def __init__(
        self,
        ks: int,
        mp: jmp.Policy,
        dino_name: str = "facebook/dino-vits8",
        dino_ckpt_path: str | None = None,
        key_depths=(2, 5, 8, 11),
        official_raev2: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        # This value is autamatically set with nnx.Module.{train(), eval()}. But nnx.SpectralNorm
        # takes an `update_stats` argument for training/eval. We store this dummy variable here
        # to track the training/eval state instead. By default, it's on training state
        self.use_running_average = False

        self.mp = mp
        self.key_depths = key_depths
        self.official_raev2 = official_raev2
        if official_raev2:
            self.dino = RAEv2DinoS8(
                dino_ckpt_path,
                resolution=224,
                dtype=mp.compute_dtype,
            )
        else:
            from pooldino.pretrained.dino import DinoViT

            self.dino = DinoViT(
                dino_name,
                resolution=224,
                dtype=jnp.float32,
                official_raev2=False,
            )

        # Infer embed_dim from DINO config
        embed_dim = self.dino.config.hidden_size
        self.heads = nnx.List(
            [
                nnx.Sequential(
                    DinoDiscBlock(
                        self.mp,
                        embed_dim,
                        1,
                        "bn",
                        official_raev2=official_raev2,
                        rngs=rngs,
                    ),
                    ResidualBlock(
                        DinoDiscBlock(
                            self.mp,
                            embed_dim,
                            ks,
                            "bn",
                            official_raev2=official_raev2,
                            rngs=rngs,
                        )
                    ),
                )
                for _ in range(len(key_depths) + 1)
            ]
        )
        self.norms = nnx.List(
            [
                EvalSafeSpectralNorm(
                    nnx.Conv(
                        embed_dim,
                        1,
                        kernel_size=1,
                        padding=0,
                        kernel_init=(
                            _torch_conv_kernel_init
                            if official_raev2
                            else nnx.initializers.lecun_normal()
                        ),
                        bias_init=(
                            _torch_conv_bias_init(embed_dim)
                            if official_raev2
                            else nnx.initializers.zeros_init()
                        ),
                        dtype=mp.compute_dtype if official_raev2 else None,
                        param_dtype=mp.param_dtype,
                        rngs=rngs,
                    ),
                    official_raev2=official_raev2,
                    rngs=rngs,
                )
                for _ in range(len(key_depths) + 1)
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through DINO discriminator.

        Args:
            x: Input images in [-1, 1] range with shape (B, H, W, C).

        Returns:
            Discriminator logits with shape (B, num_outputs).
        """
        chex.assert_rank(x, 4)
        b, h, w, c = x.shape
        is_training = not self.use_running_average

        # Resize to DINO input resolution
        if self.official_raev2:
            # ``FrozenDINONoDrop.forward`` uses F.interpolate(...,
            # mode="bilinear", align_corners=False). JAX's linear resize has
            # the same half-pixel coordinates; antialias must be disabled for
            # the 256 -> 224 downsample to match PyTorch exactly.
            x = jax.image.resize(
                x,
                (b, 224, 224, c),
                method="linear",
                antialias=False,
            )
        else:
            x = jax.image.resize(x, (b, 224, 224, c), method="bilinear")

        # Convert from [-1, 1] to ImageNet normalized space
        # This matches the reference implementation's internal normalization
        x = x * DINO_X_SCALE + DINO_X_SHIFT

        # Get intermediate activations at key_depths
        final_output, intermediate_activations = self.dino(
            x, deterministic=True, capture_layers=self.key_depths
        )

        # Remove CLS token from all activations (CLS is at index 0)
        # Reference: x[:, 1:, :].transpose(1, 2) but we keep (B, T, C) for JAX Conv
        activations = [act[:, 1:, :] for act in intermediate_activations]

        # Add final DINO output to activations (reference includes final output at index 0)
        activations.insert(0, final_output[:, 1:, :])

        outputs = []
        for head, norm, act in zip(self.heads, self.norms, activations):
            tmp = head(act)
            tmp = norm(tmp, update_stats=is_training)
            outputs.append(rearrange(tmp, "b ... -> b (...)"))

        return jnp.concatenate(outputs, axis=1)
