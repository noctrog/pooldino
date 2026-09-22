"""Pinned, checksum-verified cache for released RAEv2 model assets.

Explicit user paths remain authoritative.  When no explicit path or environment
override is supplied, assets are materialized below pooldino's shared model cache.
Downloads use Hugging Face's resumable content-addressed cache, are verified
against a fixed SHA-256, and are published to the stable path atomically.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile

from filelock import FileLock
from huggingface_hub import hf_hub_download

from pooldino.data import MODELS_CACHE_ROOT


RAEV2_MODELS_REPO_ID = "nyu-visionx/RAEv2-models"
RAEV2_MODELS_REVISION = "9770b7b980fa1875c8e6d65f226c615c0ce908a8"
RAEV2_MODELS_CACHE_ROOT = MODELS_CACHE_ROOT / "raev2"


@dataclass(frozen=True)
class RAEv2AssetSpec:
    """Immutable identity and discovery information for one released asset."""

    repo_path: str
    sha256: str
    environment_variable: str
    display_name: str
    repo_id: str = RAEV2_MODELS_REPO_ID
    revision: str = RAEV2_MODELS_REVISION

    @property
    def filename(self) -> str:
        return Path(self.repo_path).name


DINOV3_VITL16_ASSET = RAEv2AssetSpec(
    repo_path=(
        "encoders/dinov3/"
        "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    ),
    sha256="8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
    environment_variable="DINOV3_VITL16_WEIGHTS",
    display_name="RAEv2-official DINOv3-L/16 LVD-1689M checkpoint",
)

RAEV2_DINO_S8_ASSET = RAEv2AssetSpec(
    repo_path="encoders/dino/dino_vit_small_patch8_224.pth",
    sha256="6f9f986e17efd79810ef82c1f951a1b9824a8db3ee127a7a329d817a3c5d3a7b",
    environment_variable="RAEV2_DINO_S8_WEIGHTS",
    display_name="RAEv2 gan.arch.dino_ckpt_path (DINO-S/8 checkpoint)",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raev2_asset_cache_path(
    spec: RAEv2AssetSpec,
    cache_root: str | Path | None = None,
) -> Path:
    root = Path(cache_root).expanduser() if cache_root is not None else RAEV2_MODELS_CACHE_ROOT
    return root / spec.repo_path


def _validate_local_asset(
    spec: RAEv2AssetSpec,
    path: str | Path,
    *,
    source: str,
    verify_checksum: bool,
) -> Path:
    resolved = Path(path).expanduser().absolute()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{spec.display_name} was not found at {resolved} ({source})."
        )
    if verify_checksum:
        actual = sha256_file(resolved)
        if actual != spec.sha256:
            raise ValueError(
                f"{spec.display_name} checksum mismatch at {resolved}: "
                f"expected {spec.sha256}, got {actual}."
            )
    return resolved


def _download_blob(
    spec: RAEv2AssetSpec,
    cache_root: Path,
    *,
    force_download: bool,
    local_files_only: bool,
) -> Path:
    """Download through HF's resumable cache without publishing the stable path."""
    hub_cache = cache_root / ".hub"
    downloaded = hf_hub_download(
        repo_id=spec.repo_id,
        repo_type="model",
        filename=spec.repo_path,
        revision=spec.revision,
        cache_dir=hub_cache,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    # Hub snapshot entries can be relative symlinks into its blob store. Resolve
    # before hard-linking so the stable cache path never inherits that symlink.
    path = Path(downloaded).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Hugging Face returned a missing path for {spec.display_name}: {path}."
        )
    return path


def _publish_blob_atomically(blob: Path, destination: Path) -> None:
    """Publish a verified blob without exposing a partial destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        try:
            # The private HF cache and stable cache normally share a filesystem,
            # so a hard link avoids storing a second 1.2 GB DINOv3 copy.
            os.link(blob, temporary)
        except OSError:
            shutil.copyfile(blob, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_cached_asset(
    spec: RAEv2AssetSpec,
    *,
    cache_root: str | Path | None,
    verify_checksum: bool,
    force_download: bool,
    local_files_only: bool,
) -> Path:
    root = (
        Path(cache_root).expanduser().absolute()
        if cache_root is not None
        else RAEV2_MODELS_CACHE_ROOT.expanduser().absolute()
    )
    destination = root / spec.repo_path
    lock_path = root / ".locks" / f"{spec.sha256}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(lock_path):
        corrupt_destination = False
        if destination.is_file() and not force_download:
            if not verify_checksum or sha256_file(destination) == spec.sha256:
                return destination
            corrupt_destination = True
            logging.warning(
                "Ignoring corrupt cached %s at %s and downloading the pinned asset.",
                spec.display_name,
                destination,
            )

        blob = _download_blob(
            spec,
            root,
            force_download=force_download or corrupt_destination,
            local_files_only=local_files_only,
        )
        if verify_checksum and sha256_file(blob) != spec.sha256:
            # A damaged private HF blob can otherwise be returned repeatedly.
            # Force exactly one refresh, but never replace the old stable path
            # until a verified blob is ready.
            blob = _download_blob(
                spec,
                root,
                force_download=True,
                local_files_only=local_files_only,
            )
            actual = sha256_file(blob)
            if actual != spec.sha256:
                raise ValueError(
                    f"Downloaded {spec.display_name} checksum mismatch: "
                    f"expected {spec.sha256}, got {actual}."
                )

        _publish_blob_atomically(blob, destination)
        return destination


def resolve_raev2_asset(
    spec: RAEv2AssetSpec,
    explicit_path: str | Path | None = None,
    *,
    persisted_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    verify_checksum: bool = True,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    """Resolve an exact RAEv2 asset with strict override precedence.

    Explicit and environment paths are authoritative and fail immediately when
    missing or corrupt.  A persisted checkpoint path is only a relocatable hint:
    if it is no longer valid, resolution continues through the shared cache.
    """
    if explicit_path is not None:
        return _validate_local_asset(
            spec,
            explicit_path,
            source="explicit path",
            verify_checksum=verify_checksum,
        )

    environment_path = os.environ.get(spec.environment_variable)
    if environment_path:
        return _validate_local_asset(
            spec,
            environment_path,
            source=spec.environment_variable,
            verify_checksum=verify_checksum,
        )

    if persisted_path is not None:
        try:
            return _validate_local_asset(
                spec,
                persisted_path,
                source="persisted checkpoint hint",
                verify_checksum=verify_checksum,
            )
        except (FileNotFoundError, ValueError) as error:
            logging.warning("%s Falling back to the shared cache.", error)

    return _ensure_cached_asset(
        spec,
        cache_root=cache_root,
        verify_checksum=verify_checksum,
        force_download=force_download,
        local_files_only=local_files_only,
    )


__all__ = [
    "DINOV3_VITL16_ASSET",
    "RAEV2_DINO_S8_ASSET",
    "RAEV2_MODELS_CACHE_ROOT",
    "RAEV2_MODELS_REPO_ID",
    "RAEV2_MODELS_REVISION",
    "RAEv2AssetSpec",
    "raev2_asset_cache_path",
    "resolve_raev2_asset",
    "sha256_file",
]
