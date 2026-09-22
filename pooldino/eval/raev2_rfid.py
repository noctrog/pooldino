"""Compute reconstruction FID with RAEv2's torch-fidelity backend.

This intentionally mirrors ``src/eval/fid.py`` in nanovisionx/RAEv2 commit
``8a0d238f8dc3b261aba98b217f6c79c0182e8e94``.  Inputs are uint8 NHWC NPZ
files and are compared directly rather than against precomputed ADM moments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro


@dataclass
class Config:
    reference_path: Path
    reconstruction_path: Path
    batch_size: int = 128
    device: Literal["auto", "cpu", "cuda"] = "auto"
    output: Path | None = None

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_experiment: str | None = None
    wandb_commit: str | None = None
    wandb_run_name: str | None = None
    wandb_prefix: str = "eval"
    wandb_step: int | None = None


def _load_uint8_nhwc(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "arr_0" not in archive:
            raise KeyError(f"{path} does not contain the required 'arr_0' array.")
        array = np.asarray(archive["arr_0"])
    if array.dtype != np.uint8:
        raise TypeError(f"{path} must contain uint8 images, got {array.dtype}.")
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{path} must have uint8 NHWC RGB shape, got {array.shape}.")
    return array


def _align_reference_cardinality(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """Mirror RAEv2's one-sided reference alignment before rFID.

    The released evaluator never truncates reconstructions.  When counts
    differ it slices only the reference array to the reconstruction count;
    torch-fidelity still accepts the unequal case when the reference is the
    shorter input.
    """
    if len(reference) == 0 or len(reconstruction) == 0:
        raise ValueError("rFID inputs must each contain at least one image.")
    if len(reference) != len(reconstruction):
        print(
            "[Eval] Aligning ref to recon size: "
            f"{len(reference)} -> {len(reconstruction)}"
        )
        reference = reference[: len(reconstruction)]
    return reference


def _run_id(cfg: Config) -> str | None:
    if cfg.wandb_run_id is not None:
        return cfg.wandb_run_id
    if cfg.wandb_experiment is not None and cfg.wandb_commit is not None:
        return f"{cfg.wandb_experiment}-{cfg.wandb_commit}"[:128]
    return None


def _log_wandb(cfg: Config, value: float) -> None:
    if cfg.wandb_project is None:
        return
    run_id = _run_id(cfg)
    if run_id is None:
        raise ValueError(
            "--wandb-project requires --wandb-run-id, or both "
            "--wandb-experiment and --wandb-commit."
        )
    import wandb

    init_kwargs = {
        "project": cfg.wandb_project,
        "id": run_id,
        "resume": "allow",
    }
    if cfg.wandb_entity is not None:
        init_kwargs["entity"] = cfg.wandb_entity
    if cfg.wandb_run_name is not None:
        init_kwargs["name"] = cfg.wandb_run_name[:128]
    run = wandb.init(**init_kwargs)
    prefix = cfg.wandb_prefix.rstrip("/")
    payload = {
        f"{prefix}/rfid": value,
        f"{prefix}/rfid_raev2_torch_fidelity": value,
        f"{prefix}/rfid_reference": str(cfg.reference_path),
        f"{prefix}/rfid_reconstructions": str(cfg.reconstruction_path),
    }
    if cfg.wandb_step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=cfg.wandb_step)
    run.summary.update(payload)
    wandb.finish()


def main(cfg: Config) -> None:
    try:
        import torch
        from torch.utils.data import Dataset
        from torch_fidelity import calculate_metrics
    except ImportError as error:
        raise ImportError(
            "The RAEv2 metric needs torch-fidelity. Run this module with "
            "`uv run --with=torch-fidelity python -m "
            "pooldino.eval.raev2_rfid ...`."
        ) from error

    reference = _load_uint8_nhwc(cfg.reference_path)
    reconstruction = _load_uint8_nhwc(cfg.reconstruction_path)
    reference = _align_reference_cardinality(reference, reconstruction)

    class ImageArrayDataset(Dataset):
        def __init__(self, images):
            self.images = images

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return torch.from_numpy(self.images[index]).permute(2, 0, 1)

    use_cuda = torch.cuda.is_available() if cfg.device == "auto" else cfg.device == "cuda"
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but PyTorch cannot access CUDA.")
    metrics = calculate_metrics(
        input1=ImageArrayDataset(reference),
        input2=ImageArrayDataset(reconstruction),
        batch_size=cfg.batch_size,
        fid=True,
        cuda=use_cuda,
    )
    value = float(metrics["frechet_inception_distance"])
    result = {
        "rfid": value,
        "backend": "torch-fidelity",
        "num_samples": len(reconstruction),
        "num_reference_samples": len(reference),
        "num_reconstruction_samples": len(reconstruction),
        "reference_path": str(cfg.reference_path),
        "reconstruction_path": str(cfg.reconstruction_path),
    }
    output = cfg.output or cfg.reconstruction_path.with_name("raev2_rfid.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    _log_wandb(cfg, value)


if __name__ == "__main__":
    main(tyro.cli(Config))
