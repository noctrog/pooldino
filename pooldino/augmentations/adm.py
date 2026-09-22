"""ADM-style center crop augmentation.

This module provides the standard center crop preprocessing used by ADM,
DiT, DDT and other diffusion models for ImageNet evaluation.

Reference:
https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
"""

from dataclasses import dataclass

import albumentations as A
import cv2
import grain.python as grain
import numpy as np
from PIL import Image

from pooldino.data import DataConfig, seed_albumentations


def adm_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Center crop implementation from ADM (guided-diffusion).

    This is the standard preprocessing used by ADM, DiT, DDT and other
    diffusion models for ImageNet evaluation.

    Args:
        image: Input image as numpy array (H, W, C).
        size: Target size for the square crop.

    Returns:
        Center-cropped image as numpy array (size, size, C).
    """
    if image.shape[:2] == (size, size):
        return image

    pil_image = Image.fromarray(image)

    # Downsample by 2 while image is >= 2x target size (faster than single large resize)
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    # Resize so shortest side equals target size
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # Center crop to target size
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - size) // 2
    crop_x = (arr.shape[1] - size) // 2
    return arr[crop_y : crop_y + size, crop_x : crop_x + size]


class ADMCenterCrop(A.ImageOnlyTransform):
    """Albumentations wrapper for ADM-style center crop.

    This transform applies the standard ADM center crop preprocessing,
    which iteratively downsamples large images before the final resize
    for better quality.

    Args:
        size: Target size for the square crop.
        p: Probability of applying the transform.
    """

    def __init__(self, size: int, p: float = 1.0):
        super().__init__(p=p)
        self.size = size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return adm_center_crop(img, self.size)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("size",)


@dataclass
class ADMCenterCropConfig:
    """Configuration for ADM center crop augmentations."""

    crop_size: int = 256
    """Target size for the ADM center crop."""
    output_size: int | None = None
    """If specified, resize to this size after cropping (e.g., 224 for DINO)."""
    horizontal_flip: bool = False
    """Whether to apply random horizontal flip."""


class ADMCenterCropAugmentations(grain.RandomMapTransform):
    """ADM-style center crop augmentation for grain dataloaders.

    This augmentation applies:
    1. ADM center crop to crop_size (default 256)
    2. Optional resize to output_size (e.g., 224 for DINO)
    3. Optional horizontal flip
    4. ImageNet normalization

    Args:
        cfg: Configuration for the augmentation.
        data_cfg: Data configuration with normalization parameters.
    """

    def __init__(self, cfg: ADMCenterCropConfig, data_cfg: DataConfig):
        super().__init__()
        transforms = [ADMCenterCrop(cfg.crop_size)]

        if cfg.output_size is not None and cfg.output_size != cfg.crop_size:
            transforms.append(A.Resize(cfg.output_size, cfg.output_size, interpolation=cv2.INTER_LINEAR))

        if cfg.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))

        transforms.append(A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std))
        self.transforms = A.Compose(transforms)

    def random_map(self, element: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        seed_albumentations(rng, self.transforms)
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}
