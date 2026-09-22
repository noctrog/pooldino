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
