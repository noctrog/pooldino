"""RAE augmentation utilities.

This module provides augmentations commonly used for RAE and diffusion model training,
including ADM-style center cropping which is the standard preprocessing for
ImageNet evaluation (FID/sFID/IS).
"""

from pooldino.augmentations.adm import (
    adm_center_crop,
    ADMCenterCrop,
    ADMCenterCropConfig,
    ADMCenterCropAugmentations,
)
from pooldino.augmentations.decoder import (
    RAEDecoderAugConfig,
    RAEDecoderTrainAugmentations,
    RAEDecoderValAugmentations,
)

__all__ = [
    "adm_center_crop",
    "ADMCenterCrop",
    "ADMCenterCropConfig",
    "ADMCenterCropAugmentations",
    "RAEDecoderAugConfig",
    "RAEDecoderTrainAugmentations",
    "RAEDecoderValAugmentations",
]
