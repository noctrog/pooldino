"""Shared configuration/helpers for the PoolDINO paper implementation."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Literal

@dataclass
class GeneratorOptimConfig:
    epochs: int = 800
    batch_size: int = 1024
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    lr_start: float = 2e-4
    lr_peak: float = 2e-4
    lr_final: float = 2e-5
    warmup_epochs: int = 40
    ema: float = 0.9995
    lr_schedule: Literal["constant", "warmup_cosine", "wsd", "linear_decay"] = "constant"
    # The historical generator path used Optax's implicit AdamW default
    # (1e-4).  Persist it explicitly so old checkpoints keep their behaviour,
    # while exact RAEv2 experiments can request weight_decay=0.
    optimizer: Literal["adamw", "gmuon"] = "adamw"
    weight_decay: float = 1e-4
    momentum: float = 0.95
    nesterov: bool = True
    # When set, ``linear_decay`` reaches ``lr_final`` at this epoch and holds
    # it for the rest of training.  RAEv2 holds the peak LR through epoch 25,
    # decays through epoch 50, then trains at the final LR through epoch 80.
    decay_end_epoch: int | None = None

