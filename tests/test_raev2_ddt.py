"""Parity and rectangular-grid tests for the released RAEv2 DDT port."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pytest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp

from pooldino.models.raev2_ddt import (
    RAEv2DDT,
    RAEv2DDTConfig,
    RAEv2DDTDecoderBlock,
    RAEv2DDTEncoderBlock,
    RAEv2NormAttention,
    RAEv2RotaryEmbedding,
)
from pooldino.models.transformer import set_attn_implementation


MP = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)


@pytest.fixture(autouse=True)
def single_device_model_mesh():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    with jax.set_mesh(mesh):
        yield


def _small_config(grid=(2, 2), **overrides):
    values = dict(
        input_grid=grid,
        in_channels=3,
        encoder_dim=16,
        decoder_dim=16,
        encoder_depth=2,
        decoder_depth=1,
        encoder_heads=4,
        decoder_heads=4,
        mlp_ratio=2.0,
        num_classes=10,
        num_time_tokens=4,
        num_class_tokens=8,
        base_model_depth=1,
    )
    values.update(overrides)
    return RAEv2DDTConfig(**values)


def _official_rope_angles(head_dim, height, width, cond_len, theta=10_000.0):
    """Literal NumPy translation of released ``model_utils.RoPE``."""
    half_dim = head_dim // 2
    frequencies = 1.0 / (
        theta ** (np.arange(0, half_dim, 2, dtype=np.float32) / half_dim)
    )
    row = np.outer(np.arange(height, dtype=np.float32), frequencies)
    col = np.outer(np.arange(width, dtype=np.float32), frequencies)
    visual = np.concatenate(
        (
            np.broadcast_to(row[:, None], (height, width, frequencies.size)),
            np.broadcast_to(col[None], (height, width, frequencies.size)),
        ),
        axis=-1,
    ).reshape(height * width, half_dim)
    condition = np.zeros((cond_len, half_dim), dtype=np.float32)
    return np.repeat(np.concatenate((visual, condition), axis=0), 2, axis=-1)


def test_released_imagenet_defaults_are_encoded_in_config():
    cfg = RAEv2DDTConfig()

    assert cfg.input_grid == (16, 16)
    assert cfg.in_channels == 1024
    assert (cfg.encoder_dim, cfg.decoder_dim) == (1440, 2048)
    assert (cfg.encoder_depth, cfg.decoder_depth) == (28, 2)
    assert (cfg.encoder_heads, cfg.decoder_heads) == (20, 16)
    assert cfg.mlp_ratio == 4.0
    assert (cfg.num_time_tokens, cfg.num_class_tokens) == (4, 8)
    assert cfg.base_model_depth == 8


@pytest.mark.parametrize("grid", [(3, 3), (2, 4), (4, 2)])
def test_rope_matches_released_square_formula_and_rectangular_extension(grid):
    height, width = grid
    head_dim = 8
    cond_len = 12
    rope = RAEv2RotaryEmbedding(head_dim, grid, cond_len)
    expected = _official_rope_angles(head_dim, height, width, cond_len)

    np.testing.assert_allclose(np.asarray(rope.freqs_cos), np.cos(expected), atol=1e-7)
    np.testing.assert_allclose(np.asarray(rope.freqs_sin), np.sin(expected), atol=1e-7)
    # The released model gives all four time and eight label tokens zero
    # angles, hence RoPE is exactly the identity on the condition suffix.
    np.testing.assert_array_equal(np.asarray(rope.freqs_cos)[-cond_len:], 1.0)
    np.testing.assert_array_equal(np.asarray(rope.freqs_sin)[-cond_len:], 0.0)


def test_rectangular_rope_preserves_row_major_2d_coordinates():
    rope_2x4 = RAEv2RotaryEmbedding(8, (2, 4))
    rope_4x2 = RAEv2RotaryEmbedding(8, (4, 2))

    # Token 2 is (row=0,col=2) in 2x4, but (row=1,col=0) in 4x2.
    # A flattened 1-D RoPE would make these entries equal.
    assert not np.allclose(
        np.asarray(rope_2x4.freqs_cos)[2],
        np.asarray(rope_4x2.freqs_cos)[2],
    )


def test_encoder_is_plain_and_decoder_alone_has_zero_adaln():
    encoder = RAEv2DDTEncoderBlock(16, 4, 2.0, MP, rngs=nnx.Rngs(0))
    decoder = RAEv2DDTDecoderBlock(16, 4, 2.0, MP, rngs=nnx.Rngs(1))

    assert not hasattr(encoder, "adaln_modulation")
    assert hasattr(decoder, "adaln_modulation")
    np.testing.assert_array_equal(decoder.adaln_modulation.layers[1].kernel, 0.0)
    np.testing.assert_array_equal(decoder.adaln_modulation.layers[1].bias, 0.0)
    # Q/K normalization and distinct released q/k/v linears are mandatory.
    assert hasattr(encoder.attn, "q_norm") and hasattr(encoder.attn, "k_norm")
    assert encoder.attn.q is not encoder.attn.k is not encoder.attn.v


def test_attention_backend_switch_reaches_every_raev2_block():
    model = RAEv2DDT(_small_config(), MP, rngs=nnx.Rngs(11))
    attention_modules = [
        module
        for _, module in nnx.iter_modules(model)
        if isinstance(module, RAEv2NormAttention)
    ]

    assert len(attention_modules) == 3
    assert all(module.implementation == "xla" for module in attention_modules)

    set_attn_implementation(model, "cudnn")

    assert all(module.implementation == "cudnn" for module in attention_modules)
    assert all(
        module._get_attn_fn().keywords["implementation"] == "cudnn"
        for module in attention_modules
    )


@pytest.mark.parametrize("grid", [(2, 2), (2, 4), (4, 2)])
def test_forward_supports_square_and_rectangular_latent_grids(grid):
    cfg = _small_config(grid)
    model = RAEv2DDT(cfg, MP, rngs=nnx.Rngs(2))
    token_count = grid[0] * grid[1]
    x = jax.random.normal(jax.random.key(3), (2, token_count, cfg.in_channels))

    output = model(
        x,
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        # num_classes is the released null-label embedding.
        jnp.asarray([1, cfg.num_classes], dtype=jnp.int32),
        train=False,
        return_base_model=True,
    )

    assert set(output) == {"x", "base_x"}
    assert output["x"].shape == x.shape
    assert output["base_x"].shape == x.shape
    # Released output and internal-guidance heads are both zero initialized.
    np.testing.assert_array_equal(output["x"], 0.0)
    np.testing.assert_array_equal(output["base_x"], 0.0)
    assert model.encoder_rope.freqs_cos.shape[0] == token_count + 12
    assert model.decoder_rope.freqs_cos.shape[0] == token_count


def test_nontrivial_patchify_and_unpatchify_match_released_axis_orders():
    cfg = _small_config(
        (4, 6),
        encoder_patch_size=(2, 3),
        decoder_patch_size=(2, 3),
    )
    model = RAEv2DDT(cfg, MP, rngs=nnx.Rngs(4))
    x = jnp.arange(2 * 24 * 3, dtype=jnp.float32).reshape(2, 24, 3)

    image = x.reshape(2, 4, 6, 3)
    patches = model._patchify(x, cfg.encoder_patch_size)
    expected_patch_embed_input = np.asarray(image).reshape(2, 2, 2, 2, 3, 3)
    expected_patch_embed_input = expected_patch_embed_input.transpose(0, 1, 3, 5, 2, 4)
    expected_patch_embed_input = expected_patch_embed_input.reshape(2, 4, 18)
    np.testing.assert_array_equal(patches, expected_patch_embed_input)

    # The final projection is ordered [patch_h, patch_w, channel], exactly as
    # released DDT.unpatchify's reshape(..., p, p, c).
    output_patches = jnp.asarray(
        np.asarray(image)
        .reshape(2, 2, 2, 2, 3, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(2, 4, 18)
    )
    restored = model._unpatchify(
        output_patches, cfg.encoder_patch_size, model.encoder_grid
    )

    np.testing.assert_array_equal(restored, x)


def test_reference_self_repa_projector_is_optional_and_uses_visual_tokens_only():
    cfg = _small_config((2, 4))
    model = RAEv2DDT(
        cfg,
        MP,
        rngs=nnx.Rngs(5),
        self_repa_layer=1,
        self_repa_target_grid=(2, 4),
    )
    output = model(
        jnp.zeros((2, 8, 3), dtype=jnp.float32),
        jnp.asarray([0.2, 0.8], dtype=jnp.float32),
        jnp.asarray([1, 2], dtype=jnp.int32),
        train=False,
        return_self_repa=True,
    )

    assert output["self_repa"].shape == (2, 8, 3)
    with pytest.raises(ValueError, match="no self-REPA projector"):
        RAEv2DDT(cfg, MP, rngs=nnx.Rngs(6))(
            jnp.zeros((1, 8, 3), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            return_self_repa=True,
        )


def test_compressed_self_repa_expands_to_the_frozen_source_grid():
    cfg = _small_config((2, 4))
    parity_model = RAEv2DDT(cfg, MP, rngs=nnx.Rngs(7))
    assert not hasattr(parity_model, "self_repa_projector")

    model = RAEv2DDT(
        cfg,
        MP,
        rngs=nnx.Rngs(8),
        self_repa_layer=1,
        self_repa_target_grid=(4, 8),
    )
    output = model(
        jnp.zeros((2, 8, 3), dtype=jnp.float32),
        jnp.asarray([0.2, 0.8], dtype=jnp.float32),
        jnp.asarray([1, 2], dtype=jnp.int32),
        train=False,
        return_self_repa=True,
    )

    assert output["self_repa"].shape == (2, 32, 3)
    with pytest.raises(ValueError, match="no self-REPA projector"):
        parity_model(
            jnp.zeros((1, 8, 3), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            return_self_repa=True,
        )


def test_no_base_ablation_omits_internal_guidance_parameter_tree():
    cfg = _small_config((2, 4), base_model_depth=None)
    model = RAEv2DDT(cfg, MP, rngs=nnx.Rngs(9))

    assert not hasattr(model, "base_final_layer")
    output = model(
        jnp.zeros((1, 8, 3), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        train=False,
    )
    assert set(output) == {"x"}
    with pytest.raises(ValueError, match="no internal-guidance head"):
        model(
            jnp.zeros((1, 8, 3), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            return_base_model=True,
        )


def test_configuration_rejects_flattened_or_invalid_geometry():
    with pytest.raises(ValueError, match="divisible by four"):
        _small_config(encoder_dim=24, encoder_heads=4)
    with pytest.raises(ValueError, match="must divide input_grid"):
        _small_config((3, 4), encoder_patch_size=2, decoder_patch_size=2)
    with pytest.raises(ValueError, match="same token grid"):
        _small_config((4, 4), encoder_patch_size=1, decoder_patch_size=2)


def test_official_state_dict_conversion_and_layerwise_numerical_parity():
    torch = pytest.importorskip("torch")
    from pooldino.pretrained.raev2_ddt import (
        assert_raev2_ddt_full_checkpoint_parity,
        assert_raev2_ddt_parity,
        compare_raev2_ddt_torch_jax,
        convert_raev2_ddt_state_dict,
    )
    reference_path = Path(__file__).with_name("torch_raev2_ddt_reference.py")
    spec = importlib.util.spec_from_file_location("torch_raev2_ddt_reference", reference_path)
    assert spec is not None and spec.loader is not None
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    TinyOfficialRAEv2DDT = reference_module.TinyOfficialRAEv2DDT

    torch.manual_seed(9)
    official = TinyOfficialRAEv2DDT()
    # A freshly initialized official DDT has zero output/decoder AdaLN heads.
    # Nonzero synthetic trained weights exercise every mapped computation.
    with torch.no_grad():
        for parameter in official.parameters():
            parameter.normal_(mean=0.0, std=0.05)

    cfg = _small_config(
        (2, 2),
        encoder_dim=16,
        decoder_dim=32,
        encoder_heads=4,
        decoder_heads=8,
    )
    model = RAEv2DDT(cfg, MP, rngs=nnx.Rngs(10))
    jax_parameter_count = sum(
        int(value.size) for value in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    torch_parameter_count = sum(
        parameter.numel() for parameter in official.parameters() if parameter.requires_grad
    )
    assert jax_parameter_count == torch_parameter_count
    report = convert_raev2_ddt_state_dict(
        {"ema": official.state_dict(), "epoch": 80}, model
    )

    assert report.loaded_keys
    assert report.unexpected_keys == ()
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 2, 2, 3), dtype=np.float32)
    t = np.asarray([0.2, 0.8], dtype=np.float32)
    labels = np.asarray([1, 10], dtype=np.int64)
    stats = compare_raev2_ddt_torch_jax(official, model, x, t, labels)

    assert {
        "encoder_input",
        "encoder.0",
        "encoder.1",
        "condition",
        "decoder.0",
        "final_tokens",
        "base_final_tokens",
        "output",
        "base_output",
    } <= set(stats)
    assert jax.config.jax_default_matmul_precision == "highest"
    for statistic in stats.values():
        assert statistic.p99_abs <= statistic.max_abs
        assert statistic.rmse >= 0.0
        assert statistic.reference_rms >= 0.0
        assert statistic.nrmse >= 0.0
        assert -1.0 <= statistic.cosine_similarity <= 1.0 + 1e-12
    assert_raev2_ddt_parity(stats, atol=3e-5)
    assert_raev2_ddt_full_checkpoint_parity(
        stats,
        hidden_nrmse_atol=3e-5,
        final_nrmse_atol=3e-5,
        hidden_cosine_min=0.99999,
        final_cosine_min=0.99999,
        final_atol=3e-5,
    )

    # One sparse hidden outlier remains a useful diagnostic, but should not
    # reject a checkpoint whose distributional stage agreement is excellent.
    sparse_hidden_outlier = dict(stats)
    sparse_hidden_outlier["encoder.1"] = replace(
        sparse_hidden_outlier["encoder.1"],
        max_abs=10.0,
    )
    assert_raev2_ddt_full_checkpoint_parity(
        sparse_hidden_outlier,
        hidden_nrmse_atol=3e-5,
        final_nrmse_atol=3e-5,
        hidden_cosine_min=0.99999,
        final_cosine_min=0.99999,
        final_atol=3e-5,
    )

    broad_output_drift = dict(stats)
    broad_output_drift["output"] = replace(
        broad_output_drift["output"],
        nrmse=1e-2,
    )
    with pytest.raises(AssertionError, match="output: nrmse"):
        assert_raev2_ddt_full_checkpoint_parity(
            broad_output_drift,
            hidden_nrmse_atol=3e-5,
            final_nrmse_atol=3e-5,
            hidden_cosine_min=0.99999,
            final_cosine_min=0.99999,
            final_atol=3e-5,
        )

    # DiTwDDTHead (without the IG subclass) has the same tree minus the base
    # final layer. The converter must accept that ablation strictly.
    no_base_state = {
        key: value
        for key, value in official.state_dict().items()
        if not key.startswith("base_final_layer.")
    }
    no_base_cfg = _small_config(
        (2, 2),
        encoder_dim=16,
        decoder_dim=32,
        encoder_heads=4,
        decoder_heads=8,
        base_model_depth=None,
    )
    no_base_model = RAEv2DDT(no_base_cfg, MP, rngs=nnx.Rngs(12))
    no_base_report = convert_raev2_ddt_state_dict(
        {"model": no_base_state}, no_base_model, component="model"
    )
    assert no_base_report.unexpected_keys == ()
