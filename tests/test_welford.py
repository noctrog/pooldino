"""Tests for Welford's parallel algorithm for computing running statistics.

Verifies that the implementation in pooldino.utils.update_running_stats correctly
computes mean and variance in a numerically stable way.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pooldino.utils import update_running_stats


class TestWelfordBasic:
    """Basic tests for Welford's algorithm."""

    def test_single_batch(self):
        """Test that a single batch gives correct mean and variance."""
        # Create a simple batch with known statistics
        batch = jnp.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ])  # Shape: (3, 2, 2) - 3 samples, 2 tokens, 2 dims

        # Initialize accumulators
        count = 0
        mean = jnp.zeros((2, 2), dtype=jnp.float64)
        m2 = jnp.zeros((2, 2), dtype=jnp.float64)

        # Update with single batch
        count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))
        variance = m2 / count

        # Expected values (computed manually or with numpy)
        expected_mean = np.mean(batch, axis=0)
        expected_var = np.var(batch, axis=0)  # Population variance

        np.testing.assert_allclose(mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(variance, expected_var, atol=1e-6)
        assert count == 3

    def test_multiple_batches_equals_single_computation(self):
        """Test that processing in batches gives same result as all at once."""
        # Generate random data
        key = jax.random.key(42)
        all_data = jax.random.normal(key, (100, 4, 8))  # 100 samples, 4 tokens, 8 dims

        # Split into batches
        batch_size = 10
        batches = [all_data[i:i+batch_size] for i in range(0, 100, batch_size)]

        # Compute with Welford's algorithm
        count = 0
        mean = jnp.zeros((4, 8), dtype=jnp.float64)
        m2 = jnp.zeros((4, 8), dtype=jnp.float64)

        for batch in batches:
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        # Compute directly
        expected_mean = jnp.mean(all_data, axis=0)
        expected_var = jnp.var(all_data, axis=0)

        np.testing.assert_allclose(mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(welford_variance, expected_var, atol=1e-6)
        assert count == 100

    def test_uneven_batches(self):
        """Test with batches of different sizes."""
        key = jax.random.key(123)
        all_data = jax.random.normal(key, (37, 3, 5))  # Odd number of samples

        # Split into uneven batches
        batch_sizes = [10, 15, 7, 5]  # Sum = 37
        batches = []
        idx = 0
        for size in batch_sizes:
            batches.append(all_data[idx:idx+size])
            idx += size

        # Compute with Welford's algorithm
        count = 0
        mean = jnp.zeros((3, 5), dtype=jnp.float64)
        m2 = jnp.zeros((3, 5), dtype=jnp.float64)

        for batch in batches:
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        # Compute directly
        expected_mean = jnp.mean(all_data, axis=0)
        expected_var = jnp.var(all_data, axis=0)

        np.testing.assert_allclose(mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(welford_variance, expected_var, atol=1e-6)


class TestWelfordNumericalStability:
    """Tests for numerical stability of Welford's algorithm."""

    def test_large_values(self):
        """Test with large values that could cause overflow with naive sum."""
        # Large values that would overflow if summed naively in float32
        key = jax.random.key(0)
        large_data = jax.random.normal(key, (1000, 2, 3)) * 1e6

        # Process in batches
        count = 0
        mean = jnp.zeros((2, 3), dtype=jnp.float64)
        m2 = jnp.zeros((2, 3), dtype=jnp.float64)

        for i in range(0, 1000, 100):
            batch = large_data[i:i+100]
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        # Compare to numpy (which handles this correctly)
        expected_mean = np.mean(np.array(large_data), axis=0)
        expected_var = np.var(np.array(large_data), axis=0)

        # Use relative tolerance for large values
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(welford_variance, expected_var, rtol=1e-5)

    def test_small_variance(self):
        """Test with data that has very small variance."""
        # Data clustered tightly around a value
        key = jax.random.key(1)
        base = 1000.0
        small_var_data = base + jax.random.normal(key, (500, 4, 6)) * 1e-6

        count = 0
        mean = jnp.zeros((4, 6), dtype=jnp.float64)
        m2 = jnp.zeros((4, 6), dtype=jnp.float64)

        for i in range(0, 500, 50):
            batch = small_var_data[i:i+50]
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        expected_mean = np.mean(np.array(small_var_data), axis=0)
        expected_var = np.var(np.array(small_var_data), axis=0)

        # Mean should be very close to base value
        np.testing.assert_allclose(mean, expected_mean, atol=1e-6)
        # Variance should be small but positive
        np.testing.assert_allclose(welford_variance, expected_var, atol=1e-6)
        assert jnp.all(welford_variance >= 0)

    def test_shifted_data(self):
        """Test with data shifted far from zero."""
        # Use a moderate shift that doesn't cause float32 precision issues
        key = jax.random.key(2)
        shift = 1000.0
        noise = jax.random.normal(key, (200, 2, 4))
        shifted_data = shift + noise

        count = 0
        mean = jnp.zeros((2, 4), dtype=jnp.float64)
        m2 = jnp.zeros((2, 4), dtype=jnp.float64)

        for i in range(0, 200, 20):
            batch = shifted_data[i:i+20]
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        expected_mean = jnp.mean(shifted_data, axis=0)
        expected_var = jnp.var(shifted_data, axis=0)

        # Mean should be close to shift
        np.testing.assert_allclose(mean, expected_mean, atol=1e-4)
        # Variance should be close to 1.0 (standard normal)
        np.testing.assert_allclose(welford_variance, expected_var, atol=1e-4)


class TestWelfordEdgeCases:
    """Edge case tests for Welford's algorithm."""

    def test_single_sample_batch(self):
        """Test with batches of size 1."""
        data = jnp.array([
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ])  # 3 batches of size 1

        count = 0
        mean = jnp.zeros((1, 2), dtype=jnp.float64)
        m2 = jnp.zeros((1, 2), dtype=jnp.float64)

        for i in range(3):
            batch = data[i:i+1, :, :]  # Shape: (1, 1, 2)
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        all_data = data.reshape(3, 1, 2)
        expected_mean = jnp.mean(all_data, axis=0)
        expected_var = jnp.var(all_data, axis=0)

        np.testing.assert_allclose(mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(welford_variance, expected_var, atol=1e-6)

    def test_zero_variance_data(self):
        """Test with constant data (zero variance)."""
        constant_data = jnp.ones((50, 3, 4)) * 42.0

        count = 0
        mean = jnp.zeros((3, 4), dtype=jnp.float64)
        m2 = jnp.zeros((3, 4), dtype=jnp.float64)

        for i in range(0, 50, 10):
            batch = constant_data[i:i+10]
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        # Mean should be 42.0
        np.testing.assert_allclose(mean, 42.0, atol=1e-6)
        # Variance should be 0
        np.testing.assert_allclose(welford_variance, 0.0, atol=1e-6)

    def test_preserves_shape(self):
        """Test that output shapes match input shapes."""
        shapes_to_test = [
            (10, 2, 3),
            (5, 1, 100),
            (20, 50, 1),
            (8, 16, 32),
        ]

        for shape in shapes_to_test:
            batch_size, tokens, dims = shape
            batch = jnp.ones(shape)

            count = 0
            mean = jnp.zeros((tokens, dims), dtype=jnp.float64)
            m2 = jnp.zeros((tokens, dims), dtype=jnp.float64)

            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

            assert mean.shape == (tokens, dims), f"Mean shape mismatch for input {shape}"
            assert m2.shape == (tokens, dims), f"M2 shape mismatch for input {shape}"


class TestWelfordLargeScale:
    """Large-scale tests simulating real usage."""

    def test_imagenet_scale_simulation(self):
        """Simulate ImageNet-scale computation (1.2M samples)."""
        # Simulate with smaller scale but same pattern
        num_samples = 10000
        batch_size = 256
        tokens = 256  # Like DINO patches
        dims = 768  # Like DINO hidden size

        key = jax.random.key(42)

        count = 0
        mean = jnp.zeros((tokens, dims), dtype=jnp.float64)
        m2 = jnp.zeros((tokens, dims), dtype=jnp.float64)

        # Process in batches (simulate streaming)
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            key, subkey = jax.random.split(key)
            # Generate batch on the fly to avoid memory issues
            batch = jax.random.normal(subkey, (batch_size, tokens, dims))
            count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))

        welford_variance = m2 / count

        # Variance should be reasonable (close to 1 for standard normal)
        assert count == num_batches * batch_size
        assert mean.shape == (tokens, dims)
        assert welford_variance.shape == (tokens, dims)
        # Mean of standard normal should be close to 0
        np.testing.assert_allclose(jnp.mean(mean), 0.0, atol=0.05)
        # Variance of standard normal should be close to 1
        np.testing.assert_allclose(jnp.mean(welford_variance), 1.0, atol=0.05)


class TestWelfordConsistency:
    """Consistency tests comparing different batch orderings."""

    def test_order_independence(self):
        """Test that batch order doesn't affect final result."""
        key = jax.random.key(99)
        all_data = jax.random.normal(key, (80, 4, 6))

        # Process in forward order
        count1 = 0
        mean1 = jnp.zeros((4, 6), dtype=jnp.float64)
        m21 = jnp.zeros((4, 6), dtype=jnp.float64)
        for i in range(0, 80, 10):
            batch = all_data[i:i+10]
            count1, mean1, m21 = update_running_stats(count1, mean1, m21, batch.astype(jnp.float64))

        # Process in reverse order
        count2 = 0
        mean2 = jnp.zeros((4, 6), dtype=jnp.float64)
        m22 = jnp.zeros((4, 6), dtype=jnp.float64)
        for i in range(70, -1, -10):
            batch = all_data[i:i+10]
            count2, mean2, m22 = update_running_stats(count2, mean2, m22, batch.astype(jnp.float64))

        np.testing.assert_allclose(mean1, mean2, atol=1e-6)
        np.testing.assert_allclose(m21 / count1, m22 / count2, atol=1e-6)

    def test_different_batch_sizes_same_result(self):
        """Test that different batch sizes give same result."""
        key = jax.random.key(77)
        all_data = jax.random.normal(key, (120, 3, 5))

        results = []
        for batch_size in [10, 20, 30, 40, 60]:
            count = 0
            mean = jnp.zeros((3, 5), dtype=jnp.float64)
            m2 = jnp.zeros((3, 5), dtype=jnp.float64)
            for i in range(0, 120, batch_size):
                batch = all_data[i:i+batch_size]
                count, mean, m2 = update_running_stats(count, mean, m2, batch.astype(jnp.float64))
            results.append((mean, m2 / count))

        # All results should be identical
        ref_mean, ref_var = results[0]
        for mean, var in results[1:]:
            np.testing.assert_allclose(mean, ref_mean, atol=1e-6)
            np.testing.assert_allclose(var, ref_var, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
