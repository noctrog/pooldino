from pathlib import Path
import os
from pooldino.data.data import (
    DataConfig as DataConfig,
    DataLoaders as DataLoaders,
    EpochTruncatedIndexSampler as EpochTruncatedIndexSampler,
    RAEv2EpochSemantics as RAEv2EpochSemantics,
    create_dataloaders as create_dataloaders,
    DATASET_PRESETS as DATASET_PRESETS,
    IMAGENET_DEFAULT_MEAN as IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD as IMAGENET_DEFAULT_STD,
    get_interpolation as get_interpolation,
    raev2_steps_per_epoch as raev2_steps_per_epoch,
    raev2_total_updates as raev2_total_updates,
    seed_albumentations as seed_albumentations,
)

CACHE_FOLDER: Path = Path(os.environ.get("POOLDINO_CACHE_DIR", Path.home() / ".cache" / "pooldino")).expanduser()
HASH_FOLDER = CACHE_FOLDER / "hashes"
MODELS_CACHE_ROOT = CACHE_FOLDER / "pretrained_models"
