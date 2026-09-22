"""Generate one frozen row-orthonormal random projection for semantic probes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from scipy import linalg

from pooldino.eval.pooled_baseline_semantics import file_sha256


@dataclass
class Config:
    output_path: Path
    center_artifact: Path
    pool_hw: tuple[int, int]
    channels: int = 1024
    seed: int = 42


def generate_projection(cfg: Config) -> dict[str, object]:
    if cfg.channels <= 0 or any(value <= 0 for value in cfg.pool_hw):
        raise ValueError("channels and pool_hw must be positive.")
    center_path = cfg.center_artifact.expanduser().resolve()
    with np.load(center_path, allow_pickle=False) as values:
        if "block_mean" not in values.files:
            raise ValueError(f"Center artifact lacks block_mean: {center_path}")
        center = np.asarray(values["block_mean"], dtype=np.float32)
    input_dimension = int(np.prod(cfg.pool_hw) * cfg.channels)
    if center.shape != (input_dimension,):
        raise ValueError(
            f"Center width {center.shape} does not match pool {cfg.pool_hw} and "
            f"channels={cfg.channels}."
        )

    seed_sequence = np.random.SeedSequence([cfg.seed, *cfg.pool_hw, cfg.channels])
    rng = np.random.default_rng(seed_sequence)
    gaussian = rng.standard_normal((input_dimension, cfg.channels), dtype=np.float32)
    basis, triangular = linalg.qr(
        gaussian,
        mode="economic",
        overwrite_a=True,
        check_finite=False,
    )
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1
    basis *= signs[None, :]
    operator = np.asarray(basis.T, dtype=np.float32)
    gram = operator.astype(np.float64) @ operator.astype(np.float64).T
    orthogonality_error = float(np.max(np.abs(gram - np.eye(cfg.channels))))
    if orthogonality_error > 5e-5:
        raise ValueError(f"Generated projection orthogonality error is {orthogonality_error}.")

    output_path = cfg.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                operator=operator,
                center=center,
                pool_window=np.asarray(cfg.pool_hw, dtype=np.int32),
                seed=np.asarray(cfg.seed, dtype=np.int64),
                center_source_path=np.asarray(str(center_path)),
                center_source_sha256=np.asarray(file_sha256(center_path)),
                row_orthogonality_max_error=np.asarray(orthogonality_error),
            )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "shape": list(operator.shape),
        "pool_window": list(cfg.pool_hw),
        "seed": cfg.seed,
        "row_orthogonality_max_error": orthogonality_error,
    }


def main(cfg: Config) -> None:
    print(generate_projection(cfg), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
