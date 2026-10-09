from pathlib import Path

import pytest

from pooldino.paths import remap_artifact_path


def test_explicit_relocation(monkeypatch):
    monkeypatch.setenv("POOLDINO_ARTIFACT_PATH_MAP", '{"/old/output/decoder": "/new/output/decoders"}')
    assert remap_artifact_path('/old/output/decoder/run') == '/new/output/decoders/run'
    assert remap_artifact_path('/old/output/decoder-other/run') == '/old/output/decoder-other/run'


def test_longest_prefix(monkeypatch):
    monkeypatch.setenv("POOLDINO_ARTIFACT_PATH_MAP", '{"/old": "/new", "/old/specific": "/release"}')
    assert remap_artifact_path('/old/specific/run') == '/release/run'


def test_no_mapping_and_remote(monkeypatch, tmp_path):
    monkeypatch.delenv("POOLDINO_ARTIFACT_PATH_MAP", raising=False)
    assert remap_artifact_path(tmp_path) == str(tmp_path.resolve())
    assert remap_artifact_path('gs://bucket/run/') == 'gs://bucket/run'


@pytest.mark.parametrize('value', ['[]', '{', '{"/old": 1}', '{"relative": "/new"}'])
def test_invalid_mapping(monkeypatch, value):
    monkeypatch.setenv("POOLDINO_ARTIFACT_PATH_MAP", value)
    with pytest.raises(ValueError):
        remap_artifact_path(Path('/example'))


def test_relocation_does_not_bypass_decoder_identity(monkeypatch):
    from dataclasses import replace
    from pooldino.pooled_generator import PooledDecoderIdentity, _decoder_identity_mismatches

    monkeypatch.setenv("POOLDINO_ARTIFACT_PATH_MAP", '{"/old/output/decoder": "/new/output/decoders"}')
    expected = PooledDecoderIdentity('/old/output/decoder/run', 40032, True, 'raev2_github', 'abc')
    actual = replace(expected, checkpoint_path='/new/output/decoders/run')
    assert not _decoder_identity_mismatches(expected, actual)
    assert _decoder_identity_mismatches(expected, replace(actual, checkpoint_step=1))
    assert _decoder_identity_mismatches(expected, replace(actual, config_sha256='different'))


def test_relocation_does_not_bypass_stats_hash(monkeypatch):
    from dataclasses import replace
    from pooldino.pooled_generator import PooledLatentStatsIdentity, _latent_stats_identity_mismatches

    monkeypatch.setenv("POOLDINO_ARTIFACT_PATH_MAP", '{"/old": "/new"}')
    expected = PooledLatentStatsIdentity('/old/stats.npz', 'abc', 100, (64, 1024))
    actual = replace(expected, stats_path='/new/stats.npz')
    assert not _latent_stats_identity_mismatches(expected, actual)
    assert _latent_stats_identity_mismatches(expected, replace(actual, sha256='different'))


@pytest.mark.parametrize("name", ["1x1-80", "2x2-80", "2x2-180", "2x4-80", "4x2-80", "4x4-80"])
def test_released_artifacts_relocate_without_environment_mapping(monkeypatch, tmp_path, name):
    from dataclasses import replace
    from pooldino.checkpoints import RELEASES
    from pooldino.pooled_generator import (
        PooledDecoderIdentity, PooledLatentStatsIdentity,
        _decoder_identity_mismatches, _latent_stats_identity_mismatches,
    )

    monkeypatch.delenv("POOLDINO_ARTIFACT_PATH_MAP", raising=False)
    run = RELEASES[name].experiment
    original = tmp_path / "training" / "output" / "legacy-project" / "pooled-decoder" / run
    downloaded = tmp_path / "any-download-root" / "pooled-decoder" / run
    expected = PooledDecoderIdentity(str(original), 40032, True, 'raev2_github', 'abc')
    actual = replace(expected, checkpoint_path=str(downloaded))
    assert not _decoder_identity_mismatches(expected, actual)
    for change in [
        {"checkpoint_step": 40031}, {"use_ema": False},
        {"stage1_profile": "legacy"}, {"config_sha256": "wrong"},
        {"checkpoint_path": str(downloaded.parent / "wrong-run")},
    ]:
        assert _decoder_identity_mismatches(expected, replace(actual, **change))
    stats = PooledLatentStatsIdentity(str(original / 'pooled_latent_stats.npz'), 'hash', 100, (64, 1024))
    relocated = replace(stats, stats_path=str(downloaded / 'pooled_latent_stats.npz'))
    assert not _latent_stats_identity_mismatches(stats, relocated)
    for change in [
        {"sha256": "wrong"}, {"count": 99}, {"shape": (16, 1024)},
        {"stats_path": str(downloaded / "different.npz")},
    ]:
        assert _latent_stats_identity_mismatches(stats, replace(relocated, **change))


def test_automatic_release_mapping_is_narrow(monkeypatch, tmp_path):
    from pooldino.checkpoints import RELEASES
    from pooldino.pooled_generator import _artifact_path_identity

    monkeypatch.delenv("POOLDINO_ARTIFACT_PATH_MAP", raising=False)
    run = RELEASES["2x2-80"].experiment
    first = tmp_path / "a" / "pooled-decoder" / run
    assert _artifact_path_identity(first) != _artifact_path_identity(
        tmp_path / "b" / "pooled-decoder" / RELEASES["4x4-80"].experiment
    )
    for tail in [
        f"pooled-decoder/{run}/40032", f"pooled-decoder-other/{run}",
        "pooled-decoder/unknown-run", f"pooled-decoder/{run}/other.npz",
    ]:
        assert _artifact_path_identity(tmp_path / "a" / tail) != _artifact_path_identity(
            tmp_path / "b" / tail
        )
    remote = f"gs://bucket/pooled-decoder/{run}"
    assert _artifact_path_identity(remote) == ("uri", remote)
    assert _artifact_path_identity(first) != _artifact_path_identity(remote)
