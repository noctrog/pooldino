from types import SimpleNamespace

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from pooldino.data.data import _ImageFolderDataSource
from pooldino.metrics.linear_probe import LinearProbeConfig, train_classifier
from pooldino.eval import pooled_semantics
from pooldino.eval.pooled_semantics import (
    bind_pooled_embedding_fn,
    mean_pooled_embedding,
    pooled_semantic_metadata,
    pooled_semantic_result_key,
)


def test_bind_pooled_embedding_fn_binds_geometry_as_supported_partial(monkeypatch):
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
    def fake_embedding_batch(
        dino,
        tokenizer,
        images,
        *,
        backbone_resolution,
        num_prefix_tokens,
        layer_indices,
        aggregation,
        normalize_each,
        add_final_mean,
        post_pool_norm,
        representation_eps,
        grid_hw,
        pool_hw,
    ):
        del (
            num_prefix_tokens,
            layer_indices,
            aggregation,
            normalize_each,
            add_final_mean,
            post_pool_norm,
            representation_eps,
            grid_hw,
            pool_hw,
        )
        return dino + tokenizer + images + backbone_resolution

    monkeypatch.setattr(
        pooled_semantics,
        "pooled_embedding_batch",
        fake_embedding_batch,
    )
    restored = SimpleNamespace(
        dino=1,
        tokenizer=2,
        layer_indices=(11, 13),
        grid_hw=(16, 16),
        cfg=SimpleNamespace(
            backbone_resolution=256,
            num_prefix_tokens=5,
            post_pool_norm=False,
            pool_window=(2, 2),
            representation=SimpleNamespace(
                aggregation="mean",
                normalize_each=True,
                add_final_mean=True,
                eps=1e-5,
            ),
        ),
    )

    bound = bind_pooled_embedding_fn(restored)
    assert bound(3) == 262


def test_mean_pooled_embedding_uses_unique_pre_repeat_tokens():
    tokens = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(mean_pooled_embedding(tokens)),
        np.asarray([[2.0, 3.0], [6.0, 7.0]]),
    )


def test_imagefolder_backend_uses_sorted_class_labels_and_val_alias(tmp_path):
    for class_name, pixel_value in (("n00000002", 22), ("n00000001", 11)):
        class_dir = tmp_path / "val" / class_name
        class_dir.mkdir(parents=True)
        image = np.full((4, 5, 3), pixel_value, dtype=np.uint8)
        assert cv2.imwrite(str(class_dir / "example.JPEG"), image)

    source = _ImageFolderDataSource(str(tmp_path), "validation")
    assert source.class_names == ("n00000001", "n00000002")
    assert len(source) == 2
    assert int(source[0]["label"]) == 0
    assert source[0]["image"].shape == (4, 5, 3)


def test_mean_pooled_embedding_validates_token_shape():
    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        mean_pooled_embedding(jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="empty"):
        mean_pooled_embedding(jnp.zeros((2, 0, 3)))


def test_pooled_semantic_result_provenance():
    restored = SimpleNamespace(
        step=16,
        layer_indices=(11, 13, 15, 17, 19, 21, 23),
        latent_spec=SimpleNamespace(grid=(4, 8), feat=1024),
        cfg=SimpleNamespace(
            stage1_profile="raev2_github",
            dino_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
            backbone_resolution=256,
            pool_window=(4, 2),
        ),
    )
    metadata = pooled_semantic_metadata(
        restored,
        dataset="imagenet",
        use_ema=True,
        train_samples=1_281_167,
        validation_samples=50_000,
    )
    assert metadata["latent_grid"] == [4, 8]
    assert metadata["latent_tokens"] == 32
    assert metadata["feature_reduction"] == "mean_of_clean_pre_repeat_tokens"
    assert (
        pooled_semantic_result_key(
            "linear_probe", dataset="imagenet", step=16, use_ema=True
        )
        == "pooled_linear_probe_imagenet_mean_step16_ema"
    )


def test_linear_classifier_learns_small_dataset_with_dynamic_class_count():
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    prototypes = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    repeats = max(32, jax.device_count())
    features = jnp.repeat(prototypes, repeats, axis=0)
    labels = jnp.repeat(jnp.arange(3), repeats)
    classifier = train_classifier(
        LinearProbeConfig(epochs=90, batch_size=features.shape[0]),
        6.4,
        0.0,
        mesh,
        features,
        labels,
        num_classes=3,
    )
    assert classifier.linear.kernel.shape == (2, 3)
    predictions = jnp.argmax(classifier(features), axis=-1)
    assert float(jnp.mean(predictions == labels)) > 0.95
