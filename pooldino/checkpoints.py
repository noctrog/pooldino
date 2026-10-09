"""Published checkpoint catalog and downloads; no model/GPU imports required."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path


REPO_ID = "noctrog/pooldino"
# Pin the release so repeated downloads cannot silently switch model artifacts.
REVISION = "dab2e6df90eb673e7574ef9a1503c51c882ab476"
DEFAULT_DIRECTORY = Path("checkpoints/pooldino")
DECODER_STEP = 40032


@dataclass(frozen=True)
class CheckpointRelease:
    experiment: str
    generator_step: int
    ig_scale: float

    @property
    def decoder_directory(self) -> str:
        return f"pooled-decoder/{self.experiment}"

    @property
    def generator_directory(self) -> str:
        return f"pooled-generator/{self.experiment}"

    @property
    def patterns(self) -> tuple[str, ...]:
        return (
            f"{self.decoder_directory}/{DECODER_STEP}/*",
            f"{self.decoder_directory}/pooled_latent_stats.npz",
            f"{self.generator_directory}/{self.generator_step}/*",
        )

    @property
    def required_files(self) -> set[str]:
        decoder = f"{self.decoder_directory}/{DECODER_STEP}"
        generator = f"{self.generator_directory}/{self.generator_step}"
        files = {
            f"{decoder}/_CHECKPOINT_METADATA",
            f"{decoder}/config/metadata",
            f"{decoder}/decoder_ema/_METADATA",
            f"{decoder}/decoder_ema/manifest.ocdbt",
            f"{self.decoder_directory}/pooled_latent_stats.npz",
            f"{generator}/_CHECKPOINT_METADATA",
            f"{generator}/config/metadata",
            f"{generator}/model_ema/_METADATA",
            f"{generator}/model_ema/manifest.ocdbt",
        }
        if self.experiment.startswith("repeatconv"):
            files.update({
                f"{decoder}/tokenizer_ema/_METADATA",
                f"{decoder}/tokenizer_ema/manifest.ocdbt",
            })
        return files


RELEASES = {
    "1x1-80": CheckpointRelease("pool1x1-dinol-vitxl-raev2official-tfds", 100080, 1.75),
    "2x2-80": CheckpointRelease("repeatconv2x2-dinol-vitxl-raev2official-tfds", 100080, 1.75),
    "2x2-180": CheckpointRelease("repeatconv2x2-dinol-vitxl-raev2official-tfds", 225180, 2.0),
    "2x4-80": CheckpointRelease("repeatconv2x4-dinol-vitxl-raev2official-tfds", 100080, 2.0),
    "4x2-80": CheckpointRelease("repeatconv4x2-dinol-vitxl-raev2official-tfds", 100080, 2.0),
    "4x4-80": CheckpointRelease("repeatconv4x4-dinol-vitxl-raev2official-tfds", 100080, 2.75),
}
RELEASED_DECODERS = frozenset(release.experiment for release in RELEASES.values())


def selected_releases(models: list[str]) -> dict[str, CheckpointRelease]:
    if not models:
        raise ValueError("Select at least one model.")
    unknown = set(models).difference(RELEASES)
    if unknown:
        raise ValueError(f"Unknown or unpublished models: {', '.join(sorted(unknown))}")
    return {name: RELEASES[name] for name in models}


def download_checkpoints(
    models: list[str],
    output_dir: Path = DEFAULT_DIRECTORY,
    *,
    dry_run: bool = False,
) -> Path:
    """Fetch complete selected runs, deduplicating their shared decoder files."""
    selected = selected_releases(models)
    from huggingface_hub import HfApi, snapshot_download

    destination = output_dir.expanduser().resolve()
    patterns = sorted({pattern for release in selected.values() for pattern in release.patterns})
    required = set().union(*(release.required_files for release in selected.values()))
    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True)
    files = [
        entry for entry in info.siblings
        if any(fnmatchcase(entry.rfilename, pattern) for pattern in patterns)
    ]
    names = {entry.rfilename for entry in files}
    if missing := required.difference(names):
        raise RuntimeError(f"Incomplete checkpoint release; missing: {sorted(missing)}")
    sizes = [entry.size for entry in files]
    size_text = (
        f"{sum(sizes) / 1e9:.2f} GB"
        if all(size is not None for size in sizes)
        else "size unavailable"
    )
    print(f"Release: {REPO_ID}@{REVISION}")
    print(f"Selected: {', '.join(selected)}")
    print(f"Destination: {destination}")
    print(f"{len(files)} files, {size_text} total (already downloaded files are reused).")
    if dry_run:
        print("Dry run: no checkpoint files downloaded.")
        return destination

    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        allow_patterns=patterns,
        local_dir=destination,
    )
    if missing := sorted(name for name in names if not (destination / name).is_file()):
        raise RuntimeError(f"Download is incomplete; rerun the command. Missing: {missing}")
    print("Download complete. No artifact-path mapping is required with the current loader.")
    for name, release in selected.items():
        print(f"\n{name} (IG {release.ig_scale:g}):")
        print(f"  --generator-path {destination / release.generator_directory}")
        print(f"  --generator-step {release.generator_step}")
        print(f"  --pooled-decoder-path {destination / release.decoder_directory}")
        print(f"  --pooled-decoder-step {DECODER_STEP}")
    print(
        "\nPaper evaluation still requires prepared ImageNet validation data or an exported "
        "condition-label file. Neither is included in this checkpoint release; see README."
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--models", nargs="+", choices=tuple(RELEASES))
    selection.add_argument("--list", action="store_true", help="List published model choices.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--dry-run", action="store_true", help="Check files and size without downloading.")
    args = parser.parse_args(argv)
    if args.list:
        for name, release in RELEASES.items():
            print(f"{name:10s} generator step {release.generator_step}, IG {release.ig_scale:g}")
        print("The 300-epoch 4x4 generator is not in this release.")
        return
    download_checkpoints(args.models, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
