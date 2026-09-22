from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import pytest

from pooldino.data import MODELS_CACHE_ROOT
from pooldino.pretrained import raev2_assets
from pooldino.pretrained.raev2_assets import (
    DINOV3_VITL16_ASSET,
    RAEV2_DINO_S8_ASSET,
    RAEV2_MODELS_CACHE_ROOT,
    RAEV2_MODELS_REPO_ID,
    RAEV2_MODELS_REVISION,
    RAEv2AssetSpec,
    raev2_asset_cache_path,
    resolve_raev2_asset,
)
from pooldino.backbone import load_backbone


GOOD_BYTES = b"pinned-raev2-test-asset"
BAD_BYTES = b"corrupt"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def asset() -> RAEv2AssetSpec:
    return RAEv2AssetSpec(
        repo_id="test-org/test-repo",
        revision="a" * 40,
        repo_path="encoders/test/model.bin",
        sha256=_sha256(GOOD_BYTES),
        environment_variable="POOLDINO_TEST_RAEV2_ASSET",
        display_name="test RAEv2 asset",
    )


def _fake_hub_download(payload: bytes, calls: list[dict[str, object]]):
    def download(**kwargs):
        calls.append(kwargs)
        blob = Path(kwargs["cache_dir"]) / "fake-hub-blob.bin"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(payload)
        return str(blob)

    return download


def test_released_asset_identities_and_shared_cache_root():
    assert RAEV2_MODELS_REPO_ID == "nyu-visionx/RAEv2-models"
    assert RAEV2_MODELS_REVISION == "9770b7b980fa1875c8e6d65f226c615c0ce908a8"
    assert RAEV2_MODELS_CACHE_ROOT == MODELS_CACHE_ROOT / "raev2"
    assert DINOV3_VITL16_ASSET.repo_path.startswith("encoders/dinov3/")
    assert RAEV2_DINO_S8_ASSET.repo_path.startswith("encoders/dino/")
    assert raev2_asset_cache_path(DINOV3_VITL16_ASSET).is_relative_to(
        MODELS_CACHE_ROOT
    )


def test_first_download_is_pinned_verified_published_and_reused(
    tmp_path,
    monkeypatch,
    asset,
):
    calls: list[dict[str, object]] = []
    monkeypatch.delenv(asset.environment_variable, raising=False)
    monkeypatch.setattr(
        raev2_assets,
        "hf_hub_download",
        _fake_hub_download(GOOD_BYTES, calls),
    )

    cache_root = tmp_path / "cache"
    expected = cache_root / asset.repo_path
    first = resolve_raev2_asset(asset, cache_root=cache_root)
    second = resolve_raev2_asset(asset, cache_root=cache_root)

    assert first == expected.absolute()
    assert second == first
    assert first.read_bytes() == GOOD_BYTES
    assert len(calls) == 1
    assert calls[0]["repo_id"] == asset.repo_id
    assert calls[0]["repo_type"] == "model"
    assert calls[0]["filename"] == asset.repo_path
    assert calls[0]["revision"] == asset.revision
    assert calls[0]["force_download"] is False
    assert not list(expected.parent.glob(f".{expected.name}.*.tmp"))


def test_corrupt_stable_cache_is_force_redownloaded_and_replaced(
    tmp_path,
    monkeypatch,
    asset,
):
    cache_root = tmp_path / "cache"
    target = cache_root / asset.repo_path
    target.parent.mkdir(parents=True)
    target.write_bytes(BAD_BYTES)
    calls: list[dict[str, object]] = []
    monkeypatch.delenv(asset.environment_variable, raising=False)
    monkeypatch.setattr(
        raev2_assets,
        "hf_hub_download",
        _fake_hub_download(GOOD_BYTES, calls),
    )

    resolved = resolve_raev2_asset(asset, cache_root=cache_root)

    assert resolved == target.absolute()
    assert target.read_bytes() == GOOD_BYTES
    assert len(calls) == 1
    assert calls[0]["force_download"] is True


def test_hub_snapshot_symlink_is_materialized_as_a_stable_file(
    tmp_path,
    monkeypatch,
    asset,
):
    cache_root = tmp_path / "cache"
    monkeypatch.delenv(asset.environment_variable, raising=False)

    def download(**kwargs):
        hub_root = Path(kwargs["cache_dir"])
        blob = hub_root / "blobs" / "content"
        snapshot = hub_root / "snapshots" / asset.revision / asset.repo_path
        blob.parent.mkdir(parents=True, exist_ok=True)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(GOOD_BYTES)
        snapshot.symlink_to(blob, target_is_directory=False)
        return str(snapshot)

    monkeypatch.setattr(raev2_assets, "hf_hub_download", download)
    target = resolve_raev2_asset(asset, cache_root=cache_root)

    assert target.read_bytes() == GOOD_BYTES
    assert not target.is_symlink()


def test_bad_download_never_replaces_existing_destination(
    tmp_path,
    monkeypatch,
    asset,
):
    cache_root = tmp_path / "cache"
    target = cache_root / asset.repo_path
    target.parent.mkdir(parents=True)
    target.write_bytes(BAD_BYTES)
    calls: list[dict[str, object]] = []
    monkeypatch.delenv(asset.environment_variable, raising=False)
    monkeypatch.setattr(
        raev2_assets,
        "hf_hub_download",
        _fake_hub_download(b"still-wrong", calls),
    )

    with pytest.raises(ValueError, match="Downloaded test RAEv2 asset checksum mismatch"):
        resolve_raev2_asset(asset, cache_root=cache_root)

    assert target.read_bytes() == BAD_BYTES
    assert len(calls) == 2
    assert calls[0]["force_download"] is True
    assert calls[1]["force_download"] is True
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_download_exception_leaves_stable_destination_untouched(
    tmp_path,
    monkeypatch,
    asset,
):
    cache_root = tmp_path / "cache"
    target = cache_root / asset.repo_path
    target.parent.mkdir(parents=True)
    target.write_bytes(BAD_BYTES)
    monkeypatch.delenv(asset.environment_variable, raising=False)

    def fail(**_kwargs):
        raise OSError("network interrupted")

    monkeypatch.setattr(raev2_assets, "hf_hub_download", fail)
    with pytest.raises(OSError, match="network interrupted"):
        resolve_raev2_asset(asset, cache_root=cache_root)
    assert target.read_bytes() == BAD_BYTES


def test_explicit_and_environment_paths_are_authoritative(
    tmp_path,
    monkeypatch,
    asset,
):
    explicit = tmp_path / "explicit.bin"
    environment = tmp_path / "environment.bin"
    explicit.write_bytes(GOOD_BYTES)
    environment.write_bytes(GOOD_BYTES)
    monkeypatch.setenv(asset.environment_variable, str(environment))
    monkeypatch.setattr(
        raev2_assets,
        "hf_hub_download",
        lambda **_kwargs: pytest.fail("authoritative paths must skip download"),
    )

    assert resolve_raev2_asset(asset, explicit, cache_root=tmp_path / "cache") == explicit
    assert resolve_raev2_asset(asset, cache_root=tmp_path / "cache") == environment

    with pytest.raises(FileNotFoundError, match="explicit path"):
        resolve_raev2_asset(
            asset,
            tmp_path / "missing-explicit.bin",
            cache_root=tmp_path / "cache",
        )
    monkeypatch.setenv(asset.environment_variable, str(tmp_path / "missing-env.bin"))
    with pytest.raises(FileNotFoundError, match=asset.environment_variable):
        resolve_raev2_asset(asset, cache_root=tmp_path / "cache")


def test_invalid_persisted_hint_falls_back_to_shared_cache(
    tmp_path,
    monkeypatch,
    asset,
):
    cache_root = tmp_path / "cache"
    target = cache_root / asset.repo_path
    target.parent.mkdir(parents=True)
    target.write_bytes(GOOD_BYTES)
    monkeypatch.delenv(asset.environment_variable, raising=False)
    monkeypatch.setattr(
        raev2_assets,
        "hf_hub_download",
        lambda **_kwargs: pytest.fail("healthy cache must skip download"),
    )

    resolved = resolve_raev2_asset(
        asset,
        persisted_path=tmp_path / "old-machine" / "missing.bin",
        cache_root=cache_root,
    )
    assert resolved == target.absolute()


def test_concurrent_resolution_downloads_once(tmp_path, monkeypatch, asset):
    cache_root = tmp_path / "cache"
    calls: list[dict[str, object]] = []
    monkeypatch.delenv(asset.environment_variable, raising=False)

    def slow_download(**kwargs):
        calls.append(kwargs)
        time.sleep(0.05)
        blob = Path(kwargs["cache_dir"]) / "concurrent-blob.bin"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(GOOD_BYTES)
        return str(blob)

    monkeypatch.setattr(raev2_assets, "hf_hub_download", slow_download)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(resolve_raev2_asset, asset, cache_root=cache_root)
            for _ in range(2)
        ]
    paths = [future.result() for future in futures]

    assert paths[0] == paths[1]
    assert paths[0].read_bytes() == GOOD_BYTES
    assert len(calls) == 1


def test_hub_flags_are_forwarded(tmp_path, monkeypatch, asset):
    calls: list[dict[str, object]] = []
    monkeypatch.delenv(asset.environment_variable, raising=False)
    monkeypatch.setattr(
        raev2_assets,
        "hf_hub_download",
        _fake_hub_download(GOOD_BYTES, calls),
    )

    resolve_raev2_asset(
        asset,
        cache_root=tmp_path / "cache",
        force_download=True,
        local_files_only=True,
    )

    assert calls[0]["force_download"] is True
    assert calls[0]["local_files_only"] is True


def test_load_backbone_forwards_explicit_and_persisted_dinov3_paths(monkeypatch):
    calls = []

    class FakeDINOv3:
        def __init__(self, checkpoint_path, **kwargs):
            calls.append((checkpoint_path, kwargs))

    monkeypatch.setattr("pooldino.pretrained.dinov3.DINOv3ViTL16", FakeDINOv3)
    explicit = Path("/new-machine/explicit.pth")
    persisted = Path("/old-machine/persisted.pth")

    model = load_backbone(
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        resolution=256,
        checkpoint_path=explicit,
        persisted_checkpoint_path=persisted,
        dtype="test-dtype",
    )

    assert isinstance(model, FakeDINOv3)
    assert calls == [
        (
            explicit,
            {
                "resolution": 256,
                "persisted_path": persisted,
                "dtype": "test-dtype",
            },
        )
    ]
