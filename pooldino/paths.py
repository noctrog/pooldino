"""Explicit relocation of artifact metadata without weakening identity checks."""

import json
import os
from pathlib import Path

from pooldino.checkpoints import RELEASED_DECODERS


def released_decoder_artifact_identity(path: Path) -> tuple[str, ...] | None:
    """Identify known Hub decoder runs independently of their download root.

    Only the published run directories and their standard statistics file are
    recognized. Unknown experiments, nested paths and other filenames retain
    the existing stricter path checks. Callers still validate step, EMA, profile,
    configuration hash and statistics hash; this never edits checkpoint bytes.
    """
    suffix = ()
    if path.name == "pooled_latent_stats.npz":
        path = path.parent
        suffix = ("pooled_latent_stats.npz",)
    if path.parent.name == "pooled-decoder" and path.name in RELEASED_DECODERS:
        return ("pooldino-release", "pooled-decoder", path.name, *suffix)
    return None


def remap_artifact_path(path: Path | str) -> str:
    """Apply an optional local prefix map to a checkpoint's recorded path.

    POOLDINO_ARTIFACT_PATH_MAP is a JSON object mapping old absolute directory
    prefixes to new ones. Longest-prefix matching avoids accidental substring
    matches. This changes path comparison only, not checkpoint files or hashes.
    Remote URIs remain unchanged.
    """
    value = str(path)
    if "://" in value:
        return value.rstrip("/")
    raw_map = os.environ.get("POOLDINO_ARTIFACT_PATH_MAP", "{}")
    try:
        mapping = json.loads(raw_map)
    except json.JSONDecodeError as error:
        raise ValueError("POOLDINO_ARTIFACT_PATH_MAP must be a JSON object.") from error
    if not isinstance(mapping, dict) or not all(
        isinstance(old, str) and isinstance(new, str) and old and new
        for old, new in mapping.items()
    ):
        raise ValueError("POOLDINO_ARTIFACT_PATH_MAP must map directory strings to strings.")
    source_path = Path(value).expanduser().resolve()
    prefixes = []
    for old, new in mapping.items():
        old_path, new_path = Path(old).expanduser(), Path(new).expanduser()
        if not old_path.is_absolute() or not new_path.is_absolute():
            raise ValueError("Artifact mapping prefixes must be absolute local paths.")
        prefixes.append((old_path.resolve(), new_path.resolve()))
    for old, new in sorted(prefixes, key=lambda pair: len(pair[0].parts), reverse=True):
        if source_path.is_relative_to(old):
            return str(new / source_path.relative_to(old))
    return str(source_path)
