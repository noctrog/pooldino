"""Shared configuration/helpers for the PoolDINO paper implementation."""

from pathlib import Path

import flax.nnx as nnx

def load_backbone(
    dino_name: str,
    resolution: int | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    persisted_checkpoint_path: str | Path | None = None,
    **kwargs,
) -> nnx.Module:
    """Create a frozen backbone encoder from model name.

    The returned module exposes `.resolution` (int), `.patch_size` (int),
    `.num_prefix_tokens` (int), and `.encode(x)` / `.__call__(x)` for
    feature extraction.
    """
    if "dinov3" in dino_name:
        from pooldino.pretrained.dinov3 import DINOV3_VITL16_NAME, DINOv3ViTL16

        if dino_name != DINOV3_VITL16_NAME:
            raise ValueError(
                "The exact RAEv2 adapter currently supports only "
                f"{DINOV3_VITL16_NAME!r}, got {dino_name!r}."
            )
        return DINOv3ViTL16(
            checkpoint_path,
            resolution=resolution or 256,
            persisted_path=persisted_checkpoint_path,
            **kwargs,
        )

    if checkpoint_path is not None or persisted_checkpoint_path is not None:
        raise ValueError(
            "checkpoint paths are only supported by the explicit DINOv3 adapter; "
            f"got backbone {dino_name!r}."
        )

    from pooldino.pretrained.dinov2 import DinoWithRegisters
    return DinoWithRegisters(dino_name, resolution=resolution or 224, **kwargs)

