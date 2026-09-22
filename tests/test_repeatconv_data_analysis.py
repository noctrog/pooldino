from types import SimpleNamespace

import numpy as np

from pooldino.eval.repeatconv_data_analysis import (
    EqualImageBlockSampler,
    analyze_block_population,
    extract_nonoverlapping_blocks,
    stratified_imagefolder_indices,
)


def _identity_average_kernel(pool_hw: tuple[int, int], channels: int) -> np.ndarray:
    scale = 1.0 / np.prod(pool_hw)
    return np.broadcast_to(
        np.eye(channels, dtype=np.float64) * scale,
        (*pool_hw, channels, channels),
    ).copy()


def test_extract_nonoverlapping_blocks_preserves_row_major_cells():
    patches = np.arange(4 * 4, dtype=np.float32).reshape(1, 16, 1)
    blocks = extract_nonoverlapping_blocks(patches, (2, 2), grid_hw=(4, 4))
    expected = np.asarray(
        [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(blocks[0], expected)


def test_equal_image_sampler_respects_block_bound():
    patches = np.arange(3 * 256 * 2, dtype=np.float32).reshape(3, 256, 2)
    sampler = EqualImageBlockSampler((2, 2), max_blocks=5, max_images=3, seed=0)
    sampler.update(patches)
    values = sampler.finalize()
    assert values.shape == (5, 8)


def test_stratified_indices_balance_classes_and_are_deterministic():
    source = SimpleNamespace(
        class_names=("a", "b", "c"),
        samples=[(f"{label}-{index}", label) for label in range(3) for index in range(4)],
    )
    first = stratified_imagefolder_indices(source, 8, seed=7)
    second = stratified_imagefolder_indices(source, 8, seed=7)
    np.testing.assert_array_equal(first, second)
    counts = np.bincount([source.samples[int(index)][1] for index in first], minlength=3)
    assert counts.max() - counts.min() <= 1


def test_pca_and_average_recover_spatially_constant_population():
    rng = np.random.default_rng(3)
    channels = 3
    spatial = 4
    values = rng.normal(size=(128, channels))
    blocks = np.repeat(values[:, None, :], spatial, axis=1).reshape(128, -1)
    kernel = _identity_average_kernel((2, 2), channels)
    metrics, arrays = analyze_block_population(
        blocks,
        blocks,
        kernel,
        pca_components=3,
        pca_oversamples=2,
        pca_power_iterations=2,
        seed=0,
    )
    np.testing.assert_allclose(metrics["pca_explained_variance_fraction"], 1.0, atol=1e-6)
    np.testing.assert_allclose(metrics["learned_rowspace_variance_fraction"], 1.0, atol=1e-6)
    np.testing.assert_allclose(metrics["average_rowspace_variance_fraction"], 1.0, atol=1e-6)
    np.testing.assert_allclose(metrics["learned_pca_subspace_overlap"], 1.0, atol=1e-5)
    np.testing.assert_allclose(arrays["input_dct_frequency_variance"][0, 0], 1.0, atol=1e-6)


def test_pca_is_fit_only_on_training_blocks_and_scored_on_eval_blocks():
    values = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    zeros = np.zeros_like(values)
    fit_blocks = np.stack((values, zeros), axis=1)
    eval_blocks = np.stack((zeros, values), axis=1)
    kernel = _identity_average_kernel((1, 1), channels=2)
    metrics, _ = analyze_block_population(
        fit_blocks,
        eval_blocks,
        kernel,
        pca_components=1,
        pca_oversamples=1,
        pca_power_iterations=1,
        seed=0,
    )
    np.testing.assert_allclose(metrics["fit_pca_explained_variance_fraction"], 1.0, atol=1e-6)
    np.testing.assert_allclose(metrics["pca_explained_variance_fraction"], 0.0, atol=1e-6)
    np.testing.assert_allclose(metrics["learned_rowspace_variance_fraction"], 1.0, atol=1e-6)
