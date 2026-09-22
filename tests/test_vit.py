import jax
import jax.numpy as jnp
import flax.nnx as nnx
import jmp
import numpy as np

from pooldino.models.transformer import TransformerConfig, set_attn_implementation
from pooldino.models.vit import ViTConfig, ViTEncoder


def _mp() -> jmp.Policy:
    return jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )


def _make_vit(num_patches: int = 5, *, causal: bool = False) -> ViTEncoder:
    cfg = ViTConfig(
        transformer=TransformerConfig(
            embed_dim=4,
            num_heads=2,
            num_layers=2,
            mlp_hidden_dim=8,
            qk_norm=True,
            implementation="xla",
            causal=causal,
        ),
        patch=None,
        num_patches=num_patches,
        use_pos_embeds=False,
        input_dim=None,
        output_dim=None,
        final_norm=False,
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()).reshape(1, -1), ("data", "model"))
    with jax.set_mesh(mesh):
        return ViTEncoder(cfg, _mp(), rngs=nnx.Rngs(0))


def test_vit_cudnn_padding_is_inactive_for_xla_attention():
    vit = _make_vit(num_patches=5)
    tokens = jnp.ones((2, 5, 4), dtype=jnp.float32)

    padded, mask, did_pad = vit._maybe_pad_for_cudnn(tokens, None)

    assert did_pad is False
    assert padded.shape == tokens.shape
    assert mask is None


def test_vit_cudnn_padding_adds_masked_tail_key_for_odd_sequences():
    vit = _make_vit(num_patches=5)
    set_attn_implementation(vit, "cudnn")
    tokens = jnp.ones((2, 5, 4), dtype=jnp.float32)

    padded, mask, did_pad = vit._maybe_pad_for_cudnn(tokens, None)

    assert did_pad is True
    assert padded.shape == (2, 6, 4)
    mask_np = np.asarray(mask)
    assert mask_np.shape == (1, 1, 6, 6)
    assert not mask_np[..., :, -1].any()
    assert mask_np[..., :, :-1].all()


def test_vit_cudnn_padding_uses_causal_mask_when_possible():
    vit = _make_vit(num_patches=5, causal=True)
    set_attn_implementation(vit, "cudnn")
    tokens = jnp.ones((2, 5, 4), dtype=jnp.float32)

    padded, mask, did_pad = vit._maybe_pad_for_cudnn(tokens, None)

    assert did_pad is True
    assert padded.shape == (2, 6, 4)
    assert mask is None


def test_vit_cudnn_padding_preserves_existing_mask_entries():
    vit = _make_vit(num_patches=5)
    set_attn_implementation(vit, "cudnn")
    tokens = jnp.ones((2, 5, 4), dtype=jnp.float32)
    base_mask = jnp.ones((1, 1, 5, 5), dtype=jnp.bool_)
    base_mask = base_mask.at[..., 1, 2].set(False)

    _, mask, did_pad = vit._maybe_pad_for_cudnn(tokens, base_mask)

    assert did_pad is True
    mask_np = np.asarray(mask)
    np.testing.assert_array_equal(mask_np[..., :5, :5], np.asarray(base_mask))
    assert not mask_np[..., :, -1].any()
    assert mask_np[..., -1, :-1].all()


def test_vit_cudnn_padding_skips_even_sequences():
    vit = _make_vit(num_patches=6)
    set_attn_implementation(vit, "cudnn")
    tokens = jnp.ones((2, 6, 4), dtype=jnp.float32)

    padded, mask, did_pad = vit._maybe_pad_for_cudnn(tokens, None)

    assert did_pad is False
    assert padded.shape == tokens.shape
    assert mask is None
