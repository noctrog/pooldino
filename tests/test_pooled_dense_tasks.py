import math
from types import SimpleNamespace

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from jax.sharding import Mesh

from pooldino.data import ade20k as ade20k_data
from pooldino.data.ade20k import ADE20KSemanticDataSource
from pooldino.models.transformer import TransformerConfig
from pooldino.models.vit import ViTConfig
from pooldino.pooled_dense_decoder import (
    DenseOptimConfig,
    PooledTaskDecoder,
    PooledTaskDecoderConfig,
    _copy_rgb_decoder_vit_parameters,
    restore_frozen_pooled_source,
    unpatchify_task_values,
)
from pooldino.pooled_depth import (
    depth_sufficient_statistics,
    depth_valid_mask,
    log_depth_gradient_loss,
    resize_log_depth_in_metric_space,
    silog_loss,
    summarize_depth_statistics,
)
from pooldino.train_decoder import repeat_pooled_tokens
from pooldino.train_segmentation import (
    _masked_ce_statistics,
)
from pooldino.segmentation.configs import SegDataConfig
from pooldino.segmentation.data import (
    SegValAugmentations,
    _normalize_mask_labels,
)
from pooldino.models.decoder import RAEDecoder


def _small_task_config(
    output_channels: int = 3,
    output_patch_size: int = 1,
) -> PooledTaskDecoderConfig:
    vit = ViTConfig(
        patch=None,
        num_patches=4,
        input_dim=8,
        use_pos_embeds=True,
        pos_embed_type="sincos",
        use_cls=True,
        num_registers=0,
        latent_upsample="none",
        transformer=TransformerConfig(
            embed_dim=16,
            num_layers=2,
            num_heads=4,
            mlp_hidden_dim=32,
        ),
    )
    return PooledTaskDecoderConfig(
        vit=vit,
        output_channels=output_channels,
        output_grid=(2, 2),
        output_patch_size=output_patch_size,
    )


def test_task_decoder_requires_repeated_tokens_not_latent_interpolation():
    cfg = _small_task_config()
    cfg.vit.latent_upsample = "bilinear"
    with pytest.raises(ValueError, match="interpolation must be disabled"):
        PooledTaskDecoderConfig(
            vit=cfg.vit,
            output_channels=3,
            output_grid=(2, 2),
        )


def test_task_decoder_uses_positional_grid_and_resizes_only_its_output():
    devices = np.asarray(jax.devices()).reshape(1, -1)
    mesh = Mesh(devices, ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )
    model = PooledTaskDecoder(_small_task_config(), mp, rngs=nnx.Rngs(0))
    # Every content token is identical. Fixed per-position embeddings still let
    # the ViT produce distinct spatial outputs.
    output = np.asarray(
        model(
            jnp.ones((1, 4, 8), dtype=jnp.float32),
            output_hw=(4, 5),
            deterministic=True,
        )
    )
    assert output.shape == (1, 4, 5, 3)
    assert np.std(output) > 0


def test_patchwise_task_values_unpatchify_in_rgb_decoder_order():
    values = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
    dense = np.asarray(
        unpatchify_task_values(
            values,
            grid_hw=(2, 2),
            patch_size=2,
            output_channels=1,
        )
    )[..., 0]
    np.testing.assert_array_equal(
        dense,
        np.asarray(
            [
                [
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [8, 9, 12, 13],
                    [10, 11, 14, 15],
                ]
            ]
        ),
    )


def test_single_value_per_token_uses_direct_spatial_reshape():
    values = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    dense = np.asarray(
        unpatchify_task_values(
            values,
            grid_hw=(2, 2),
            patch_size=1,
            output_channels=3,
        )
    )
    np.testing.assert_array_equal(dense, np.asarray(values).reshape(1, 2, 2, 3))


def test_patchwise_task_decoder_emits_dense_native_grid():
    devices = np.asarray(jax.devices()).reshape(1, -1)
    jax.set_mesh(Mesh(devices, ("data", "model")))
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )
    model = PooledTaskDecoder(
        _small_task_config(output_channels=1, output_patch_size=2),
        mp,
        rngs=nnx.Rngs(0),
    )
    output = model(
        jnp.ones((1, 4, 8), dtype=jnp.float32),
        output_hw=(4, 4),
        deterministic=True,
    )
    assert output.shape == (1, 4, 4, 1)
    assert model.task_head.kernel.shape[-1] == 4


def test_depth_patch_head_unpatchifies_before_metric_resize():
    native_metric_depth = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    resized_log_depth = resize_log_depth_in_metric_space(
        jnp.log(native_metric_depth),
        output_hw=(4, 4),
    )
    assert resized_log_depth.shape == (1, 4, 4)
    resized_metric_depth = np.exp(np.asarray(resized_log_depth))
    assert resized_metric_depth.min() >= 1.0
    assert resized_metric_depth.max() <= 4.0


def test_rgb_decoder_initialization_copies_vit_but_not_task_head():
    devices = np.asarray(jax.devices()).reshape(1, -1)
    mesh = Mesh(devices, ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )
    cfg = _small_task_config(output_channels=3)
    source = RAEDecoder(
        cfg.vit,
        patch_size=16,
        num_channels=3,
        mp=mp,
        rngs=nnx.Rngs(10),
    )
    target = PooledTaskDecoder(cfg, mp, rngs=nnx.Rngs(20))
    head_before = [
        np.asarray(value).copy()
        for value in jax.tree.leaves(nnx.state(target.task_head, nnx.Param))
    ]

    _copy_rgb_decoder_vit_parameters(target, source)

    source_state = nnx.state(source, nnx.Param)
    source_state.pop("decoder_proj")
    for source_value, target_value in zip(
        jax.tree.leaves(source_state),
        jax.tree.leaves(nnx.state(target.vit, nnx.Param)),
        strict=True,
    ):
        np.testing.assert_array_equal(source_value, target_value)
    for before, after in zip(
        head_before,
        jax.tree.leaves(nnx.state(target.task_head, nnx.Param)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize(
    ("pool_hw", "unique_tokens"),
    [
        ((1, 1), 16),
        ((2, 2), 4),
        ((2, 4), 2),
        ((4, 2), 2),
        ((4, 4), 1),
    ],
)
def test_nearest_repeat_always_restores_source_grid(pool_hw, unique_tokens):
    pooled_h = 4 // pool_hw[0]
    pooled_w = 4 // pool_hw[1]
    tokens = jnp.arange(unique_tokens, dtype=jnp.float32).reshape(
        1, unique_tokens, 1
    )
    repeated = repeat_pooled_tokens(
        tokens,
        pooled_grid_hw=(pooled_h, pooled_w),
        pool_hw=pool_hw,
    )
    assert repeated.shape == (1, 16, 1)


def test_masked_cross_entropy_excludes_ignore_pixels():
    logits = jnp.asarray(
        [[[[10.0, -10.0], [-10.0, 10.0]], [[-10.0, 10.0], [10.0, -10.0]]]]
    )
    masks = jnp.asarray([[[0, 1], [-1, 1]]], dtype=jnp.int32)
    loss, _, valid_count, correct = _masked_ce_statistics(
        logits,
        masks,
        num_classes=2,
        ignore_index=-1,
    )
    assert float(valid_count) == 3.0
    assert float(correct) == 2.0
    assert float(loss) > 0.0


def test_masked_cross_entropy_excludes_padded_batch_entries():
    logits = jnp.asarray(
        [
            [[[10.0, -10.0]]],
            [[[-10.0, 10.0]]],
        ]
    )
    masks = jnp.asarray([[[0]], [[1]]], dtype=jnp.int32)
    _, _, valid_count, correct = _masked_ce_statistics(
        logits,
        masks,
        num_classes=2,
        ignore_index=-1,
        valid_batch_size=jnp.asarray(1),
    )
    assert float(valid_count) == 1.0
    assert float(correct) == 1.0


def test_ade20k_semantic_labels_normalize_and_reject_instance_rgb():
    mask = np.asarray([[0, 1, 150]], dtype=np.uint8)
    normalized = _normalize_mask_labels(
        mask,
        dataset="ade",
        raw_ignore_index=0,
        ignore_index=-1,
    )
    np.testing.assert_array_equal(normalized, [[-1, 0, 149]])
    with pytest.raises(ValueError, match="single-channel"):
        _normalize_mask_labels(
            np.zeros((2, 2, 3), dtype=np.uint8),
            dataset="ade",
            raw_ignore_index=0,
            ignore_index=-1,
        )


def test_ade20k_semantic_source_pairs_images_and_grayscale_masks(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "ADEChallengeData2016"
    image_dir = root / "images" / "training"
    annotation_dir = root / "annotations" / "training"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 255
    annotation = np.asarray(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 149, 150]],
        dtype=np.uint8,
    )
    assert cv2.imwrite(str(image_dir / "sample.jpg"), image)
    assert cv2.imwrite(str(annotation_dir / "sample.png"), annotation)
    monkeypatch.setitem(ade20k_data._EXPECTED_COUNTS, "training", 1)

    source = ADE20KSemanticDataSource(root, "train")
    equivalent_source = ADE20KSemanticDataSource(root, "training")
    sample = source[0]
    assert len(source) == 1
    assert repr(source) == repr(equivalent_source)
    assert "object at" not in repr(source)
    assert "split='training'" in repr(source)
    assert sample["image"].shape == (3, 4, 3)
    np.testing.assert_array_equal(sample["annotation"], annotation)

    rgb_annotation = np.zeros((3, 4, 3), dtype=np.uint8)
    assert cv2.imwrite(str(annotation_dir / "sample.png"), rgb_annotation)
    with pytest.raises(ValueError, match="annotations_instance"):
        source[0]


def test_ade20k_fixed_validation_preserves_full_image_and_ignores_padding():
    config = SegDataConfig(
        dataset_key="ade",
        tfds_name="ade20k_semantic",
        train_split="train",
        val_split="validation",
        mask_field="annotation",
        num_classes=150,
        ignore_index=-1,
        raw_ignore_index=0,
        mask_resolution=4,
        encoder_resolution=4,
        rrc_scale=(0.5, 2.0),
        color_jitter=False,
    )
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    annotation = np.asarray([[1, 2, 3, 4], [5, 6, 7, 150]], dtype=np.uint8)

    output = SegValAugmentations(config).map(
        {"image": image, "annotation": annotation}
    )

    assert output["image"].shape == (4, 4, 3)
    assert output["mask"].shape == (4, 4)
    valid = output["mask"][output["mask"] != -1]
    assert valid.size == annotation.size
    np.testing.assert_array_equal(np.sort(valid), [0, 1, 2, 3, 4, 5, 6, 149])


def test_perfect_log_depth_has_zero_errors_up_to_silog_epsilon():
    target = jnp.asarray([[[1.0, 2.0], [4.0, 8.0]]], dtype=jnp.float32)
    prediction = jnp.log(target)
    valid = depth_valid_mask(target, min_depth=0.1, max_depth=10.0)
    loss = silog_loss(prediction, target, valid, coefficient=0.5)
    gradient, gradient_sum, pair_count = log_depth_gradient_loss(
        prediction, target, valid
    )
    assert float(loss) == pytest.approx(1e-3, rel=1e-5)
    assert float(gradient) == pytest.approx(0.0, abs=1e-7)
    assert float(gradient_sum) == pytest.approx(0.0, abs=1e-7)
    assert float(pair_count) == 4.0


def test_depth_valid_mask_excludes_padded_batch_entries():
    target = jnp.ones((4, 2, 2), dtype=jnp.float32)
    valid = depth_valid_mask(
        target,
        min_depth=0.1,
        max_depth=10.0,
        valid_batch_size=jnp.asarray(2),
    )
    np.testing.assert_array_equal(
        np.asarray(valid[:, 0, 0]),
        np.asarray([True, True, False, False]),
    )


def test_depth_statistics_are_additive_and_summarize_exactly():
    target = jnp.asarray([[[1.0, 2.0]]], dtype=jnp.float32)
    log_prediction = jnp.log(jnp.asarray([[[1.0, 4.0]]], dtype=jnp.float32))
    valid = jnp.ones_like(target, dtype=jnp.bool_)
    stats = depth_sufficient_statistics(
        log_prediction,
        target,
        valid,
        min_depth=0.1,
        max_depth=10.0,
    )
    summary = summarize_depth_statistics(
        {name: float(value) for name, value in stats.items()},
        silog_lambda=0.5,
    )
    assert summary["abs_rel"] == pytest.approx(0.5)
    assert summary["rmse"] == pytest.approx(math.sqrt(2.0))
    assert summary["delta1"] == pytest.approx(0.5)
    assert summary["valid_pixels"] == 2.0


def test_dense_optimizer_rejects_invalid_warmup():
    with pytest.raises(ValueError, match="warmup_epochs"):
        DenseOptimConfig(epochs=2, warmup_epochs=2)


def test_source_restore_requests_tokenizer_ema_without_rgb_decoder(monkeypatch):
    calls = {}

    class FrozenModule:
        def __init__(self):
            self.was_eval = False

        def eval(self):
            self.was_eval = True

    restored = SimpleNamespace(
        decoder=None,
        step=40032,
        grid_hw=(16, 16),
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
        dino=FrozenModule(),
        tokenizer=FrozenModule(),
    )

    def fake_restore(*args, **kwargs):
        calls.update(kwargs)
        return restored

    monkeypatch.setattr(
        "pooldino.pooled_dense_decoder.restore_pooled_decoder_components",
        fake_restore,
    )
    result = restore_frozen_pooled_source(
        "checkpoint",
        mesh=None,
        mp=None,
        step=40032,
    )
    assert result is restored
    assert calls["use_ema"] is True
    assert calls["restore_decoder"] is False
    assert restored.dino.was_eval
    assert restored.tokenizer.was_eval
