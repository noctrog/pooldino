import numpy as np

from pooldino.eval.repeatconv_operator_analysis import (
    analyze_kernel,
    average_operator,
    best_scalar_relative_error,
    dct_frequency_energy,
    flatten_hwio_kernel,
    position_kernel_deviation,
)


def _average_kernel(pool_hw: tuple[int, int], channels: int) -> np.ndarray:
    pool_h, pool_w = pool_hw
    eye = np.eye(channels, dtype=np.float64) / (pool_h * pool_w)
    return np.broadcast_to(eye, (pool_h, pool_w, channels, channels)).copy()


def test_flatten_hwio_kernel_matches_explicit_convolution_block():
    kernel = np.arange(2 * 3 * 2 * 4, dtype=np.float64).reshape(2, 3, 2, 4)
    block = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)
    expected = np.einsum("hwi,hwio->o", block, kernel)
    actual = flatten_hwio_kernel(kernel) @ block.reshape(-1)
    np.testing.assert_allclose(actual, expected)


def test_average_operator_matches_spatial_channel_mean():
    block = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
    operator = average_operator((2, 2), 3, 3)
    np.testing.assert_allclose(operator @ block.reshape(-1), block.mean(axis=(0, 1)))


def test_average_kernel_has_only_dc_energy_and_identical_positions():
    kernel = _average_kernel((2, 4), 3)
    energy = dct_frequency_energy(kernel)
    np.testing.assert_allclose(energy[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(energy[1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(energy[0, 1:], 0.0, atol=1e-12)
    assert position_kernel_deviation(kernel) == 0.0


def test_average_analysis_reports_exact_subspace_and_full_rank():
    kernel = _average_kernel((2, 2), 4)
    metrics, arrays = analyze_kernel(kernel, np.zeros(4), rank_rtol=1e-12)
    assert metrics["numerical_rank"] == 4
    np.testing.assert_allclose(metrics["average_subspace_overlap"], 1.0, atol=1e-10)
    np.testing.assert_allclose(metrics["position_kernel_deviation"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["dct_dc_energy_fraction"], 1.0, atol=1e-12)
    np.testing.assert_allclose(arrays["average_principal_cosines"], 1.0, atol=1e-10)


def test_position_specific_kernel_departs_from_average_subspace():
    kernel = _average_kernel((2, 2), 3)
    kernel[0, 0] += np.diag([0.5, -0.25, 0.125])
    metrics, _ = analyze_kernel(kernel, np.zeros(3), rank_rtol=1e-12)
    assert metrics["position_kernel_deviation"] > 0
    assert metrics["dct_non_dc_energy_fraction"] > 0
    assert metrics["average_subspace_overlap"] < 1


def test_best_scalar_error_is_invariant_to_reference_scaling():
    rng = np.random.default_rng(0)
    operator = rng.normal(size=(4, 12))
    reference = rng.normal(size=(4, 12))
    np.testing.assert_allclose(
        best_scalar_relative_error(operator, reference),
        best_scalar_relative_error(operator, 7.0 * reference),
    )
