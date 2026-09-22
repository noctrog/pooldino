import math

from pooldino.training import jax_system_metrics


class FakeDevice:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats


def test_jax_system_metrics_aggregates_memory_stats():
    gib = 1024**3

    metrics = jax_system_metrics(
        devices=[
            FakeDevice(
                {
                    "bytes_in_use": 1 * gib,
                    "peak_bytes_in_use": 2 * gib,
                    "bytes_reserved": 3 * gib,
                    "bytes_limit": 4 * gib,
                }
            ),
            FakeDevice(
                {
                    "bytes_in_use": 2 * gib,
                    "peak_bytes_in_use": 3 * gib,
                    "bytes_reserved": 4 * gib,
                    "bytes_limit": 4 * gib,
                    "largest_free_block_bytes": 99 * gib,
                }
            ),
        ]
    )

    assert math.isclose(metrics["system_metrics/jax_memory_in_use_max_gib"], 2.0)
    assert math.isclose(metrics["system_metrics/jax_memory_in_use_total_gib"], 3.0)
    assert math.isclose(metrics["system_metrics/jax_memory_peak_in_use_max_gib"], 3.0)
    assert math.isclose(metrics["system_metrics/jax_memory_reserved_max_gib"], 4.0)
    assert math.isclose(metrics["system_metrics/jax_memory_utilization_max"], 0.5)
    assert math.isclose(metrics["system_metrics/jax_memory_utilization_total"], 0.375)
    assert "system_metrics/jax_memory_limit_max_gib" not in metrics
    assert "system_metrics/jax_memory_largest_free_block_max_gib" not in metrics


def test_jax_system_metrics_handles_missing_memory_stats():
    metrics = jax_system_metrics(devices=[FakeDevice(None)])

    assert metrics == {}
