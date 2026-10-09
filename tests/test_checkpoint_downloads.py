"""Checkpoint selection and download tests, without network or large weights."""

from fnmatch import fnmatchcase
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import huggingface_hub
import pytest

from pooldino.checkpoints import (
    DECODER_STEP,
    RELEASES,
    REPO_ID,
    REVISION,
    download_checkpoints,
    main,
    selected_releases,
)


@pytest.fixture
def hub(monkeypatch):
    names = set().union(*(release.required_files for release in RELEASES.values()))
    for release in RELEASES.values():
        # Optimizers and raw weights remain included, as in the full Hub runs.
        names.add(f"{release.generator_directory}/{release.generator_step}/optim/d/data")
        names.add(f"{release.decoder_directory}/{DECODER_STEP}/decoder/d/data")
    names.add("README.md")
    state = SimpleNamespace(names=names, calls=[], omit=None)

    class Api:
        def model_info(self, repo_id, *, revision, files_metadata):
            assert (repo_id, revision, files_metadata) == (REPO_ID, REVISION, True)
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename=name, size=100) for name in sorted(state.names)
            ])

    def download(**kwargs):
        state.calls.append(kwargs)
        assert kwargs["repo_id"] == REPO_ID
        assert kwargs["revision"] == REVISION
        for name in state.names:
            if name == state.omit:
                continue
            if any(fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"]):
                path = Path(kwargs["local_dir"]) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test checkpoint fixture")
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return state


def test_selected_releases_are_deduplicated():
    assert list(selected_releases(["2x2-80", "2x2-180", "2x2-80"])) == ["2x2-80", "2x2-180"]


@pytest.mark.parametrize("models", [[], ["4x4-300"], ["unknown"], ["2x2-80", "unknown"]])
def test_invalid_selections_are_rejected(models):
    with pytest.raises(ValueError):
        selected_releases(models)


def test_dry_run_does_not_download_or_create_destination(tmp_path, hub, capsys):
    output = tmp_path / "checkpoints"
    assert download_checkpoints(["2x2-80"], output, dry_run=True) == output
    assert not output.exists()
    assert not hub.calls
    assert "Dry run" in capsys.readouterr().out


def test_download_fetches_only_selected_runs_and_shares_decoders(tmp_path, hub):
    output = download_checkpoints(["2x2-80", "2x2-180", "2x2-80"], tmp_path)
    assert output == tmp_path
    assert len(hub.calls) == 1
    patterns = hub.calls[0]["allow_patterns"]
    assert len(patterns) == len(set(patterns)) == 4
    expected = {
        name for name in hub.names
        if any(fnmatchcase(name, pattern) for pattern in patterns)
    }
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    assert actual == expected
    assert not (output / RELEASES["4x4-80"].decoder_directory).exists()
    assert not (output / "README.md").exists()
    assert any("/optim/" in name for name in actual)
    assert len(list((output / "pooled-decoder").iterdir())) == 1


def test_incomplete_hub_release_fails_before_download(tmp_path, hub):
    hub.names.remove(f"{RELEASES['2x2-80'].decoder_directory}/pooled_latent_stats.npz")
    with pytest.raises(RuntimeError, match="Incomplete checkpoint release"):
        download_checkpoints(["2x2-80"], tmp_path)
    assert not hub.calls


def test_incomplete_download_is_detected(tmp_path, hub):
    hub.omit = f"{RELEASES['2x2-80'].generator_directory}/100080/optim/d/data"
    with pytest.raises(RuntimeError, match="Download is incomplete"):
        download_checkpoints(["2x2-80"], tmp_path)


def test_list_works_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "pooldino.checkpoints", "--list"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "2x2-180" in result.stdout
    assert "225180" in result.stdout


def test_cli_dry_run(tmp_path, hub):
    main(["--models", "2x4-80", "--dry-run", "--output-dir", str(tmp_path / "new")])
    assert not hub.calls


def test_cli_rejects_unpublished_model():
    with pytest.raises(SystemExit) as error:
        main(["--models", "4x4-300"])
    assert error.value.code == 2
