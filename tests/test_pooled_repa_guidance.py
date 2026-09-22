from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pooldino.guidance import (
    apply_internal_guidance,
    apply_repa_guidance,
)
from pooldino.eval import gfid_pooled_decoder_adm as pooled_eval


@pytest.mark.parametrize("official_scale", [1.0, 1.25, 1.78, 2.0])
def test_internal_guidance_matches_legacy_additive_convention(official_scale: float):
    full = jnp.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float32)
    base = jnp.asarray([[0.25, 1.5], [-2.0, 0.25]], dtype=jnp.float32)

    official = apply_internal_guidance(full, base, official_scale)
    legacy = apply_repa_guidance(full, base, official_scale - 1.0)

    np.testing.assert_allclose(official, legacy)


def test_internal_and_legacy_guidance_neutral_points_return_full_prediction():
    full = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    base = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)

    np.testing.assert_allclose(apply_internal_guidance(full, base, 1.0), full)
    np.testing.assert_allclose(apply_repa_guidance(full, base, 0.0), full)


class _ScheduleOnlyGenerator:
    cfg = SimpleNamespace(num_classes=1000)


def _record_schedule(
    monkeypatch: pytest.MonkeyPatch,
    *,
    steps: int,
    time_shift: float | None,
) -> list[tuple[float, bool]]:
    calls: list[tuple[float, bool]] = []

    def fake_integration_step(
        generator,
        state,
        t,
        dt,
        labels,
        null_labels,
        **kwargs,
    ):
        del generator, dt, labels, null_labels
        calls.append((float(t[0]), kwargs["use_internal_guidance"]))
        return state

    monkeypatch.setattr(pooled_eval, "integration_step", fake_integration_step)
    pooled_eval.sample_latents(
        _ScheduleOnlyGenerator(),
        jnp.asarray([0], dtype=jnp.int32),
        jax.random.PRNGKey(0),
        latent_shape=(1, 1),
        steps=steps,
        time_shift=time_shift,
        cfg_scale=None,
        cfg_interval=(0.0, 1.0),
        self_repa_scale=None,
        self_repa_interval=(0.0, 1.0),
        ig_scale=1.78,
        ig_interval=(0.0, 0.9),
    )
    return calls


def test_local_internal_guidance_window_is_inclusive(monkeypatch: pytest.MonkeyPatch):
    calls = _record_schedule(monkeypatch, steps=10, time_shift=None)

    assert len(calls) == 10
    assert calls[0][0] == pytest.approx(1.0)
    assert calls[-1][0] == pytest.approx(0.1)
    assert all(enabled for _, enabled in calls)


def test_released_shifted_schedule_guides_first_99_of_100_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = _record_schedule(monkeypatch, steps=100, time_shift=8.0)
    enabled = [flag for _, flag in calls]

    assert enabled == [True] * 99 + [False]
    assert calls[0][0] == pytest.approx(1.0)
    assert calls[98][0] > 0.1 > calls[99][0]


class _ConstantPredictionGenerator(nnx.Module):
    def __call__(
        self,
        state,
        t,
        labels,
        *,
        train,
        return_self_repa,
        return_base_model,
    ):
        del t, labels, train
        assert not return_self_repa
        assert return_base_model
        return {
            "x": jnp.full_like(state, 2.0),
            "base_x": jnp.full_like(state, 1.0),
        }


def test_one_step_internal_guidance_uses_official_scale():
    state = jnp.zeros((2, 1, 1), dtype=jnp.float32)
    labels = jnp.asarray([1, 2], dtype=jnp.int32)

    result = pooled_eval.integration_step(
        _ConstantPredictionGenerator(),
        state,
        jnp.ones((2,), dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        labels,
        jnp.zeros_like(labels),
        use_cfg=False,
        cfg_scale=0.0,
        use_self_repa=False,
        self_repa_scale=0.0,
        use_internal_guidance=True,
        ig_scale=1.5,
    )

    # base + 1.5 * (full - base) = 2.5. At official t=1 the
    # denominator is one, so a dt=0.25 reverse Euler update produces 0.625.
    np.testing.assert_allclose(result, np.full((2, 1, 1), 0.625, dtype=np.float32))


class _SelfRepaPredictionGenerator(nnx.Module):
    def __call__(
        self,
        state,
        t,
        labels,
        *,
        train,
        return_self_repa,
        return_base_model,
    ):
        del t, labels, train
        assert return_self_repa
        assert not return_base_model
        return {
            "x": jnp.full_like(state, 2.0),
            "self_repa": jnp.full_like(state, 3.0),
        }

    def normalize(self, value):
        return (value - 1.0) / 2.0


def test_self_repa_guidance_normalizes_the_raw_encoder_target_domain():
    state = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    labels = jnp.asarray([1], dtype=jnp.int32)

    result = pooled_eval.integration_step(
        _SelfRepaPredictionGenerator(),
        state,
        jnp.ones((1,), dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        labels,
        jnp.zeros_like(labels),
        use_cfg=False,
        cfg_scale=0.0,
        use_self_repa=True,
        self_repa_scale=1.0,
    )

    # Raw self-REPA 3 normalizes to 1; full + (full - self) = 3.
    np.testing.assert_allclose(result, 0.75, rtol=0.0, atol=1e-6)


class _CompressedSelfRepaPredictionGenerator(nnx.Module):
    def __call__(
        self,
        state,
        t,
        labels,
        *,
        train,
        return_self_repa,
        return_base_model,
    ):
        del t, labels, train
        assert return_self_repa
        assert not return_base_model
        dense = jnp.arange(0.0, 16.0, 2.0, dtype=state.dtype).reshape(1, 8, 1)
        return {
            "x": jnp.asarray([[[3.0], [5.0]]], dtype=state.dtype),
            "self_repa": jnp.broadcast_to(dense, (state.shape[0], 8, 1)),
        }

    def normalize(self, value):
        return (value - 1.0) / 2.0


class _AnisotropicMeanTokenizer(nnx.Module):
    def __call__(self, patches, *, grid_hw):
        assert grid_hw == (2, 4)
        batch, _, channels = patches.shape
        field = patches.reshape(batch, 2, 4, channels)
        pooled = field.reshape(batch, 1, 2, 2, 2, channels).mean(axis=(2, 4))
        return pooled.reshape(batch, 2, channels)


def test_compressed_self_repa_guidance_tokenizes_the_full_source_grid():
    state = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    labels = jnp.asarray([1], dtype=jnp.int32)

    result = pooled_eval.integration_step(
        _CompressedSelfRepaPredictionGenerator(),
        state,
        jnp.ones((1,), dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        labels,
        jnp.zeros_like(labels),
        use_cfg=False,
        cfg_scale=0.0,
        use_self_repa=True,
        self_repa_scale=1.0,
        self_repa_tokenizer=_AnisotropicMeanTokenizer(),
        self_repa_source_grid=(2, 4),
    )

    # The 2x2 cells pool to [5, 9], then normalize to [2, 4]. Guidance gives
    # [4, 6], and one reverse-Euler step with dt=.25 produces [1, 1.5].
    np.testing.assert_allclose(
        result,
        np.asarray([[[1.0], [1.5]]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
