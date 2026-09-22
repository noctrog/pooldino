from dataclasses import dataclass

import albumentations as A
import cv2
import grain.python as grain
import numpy as np

from pooldino.data import DataConfig, seed_albumentations
from pooldino.augmentations.adm import ADMCenterCrop, adm_center_crop


@dataclass
class RAEDecoderAugConfig:
    resize: tuple[int, int] = (384, 384)
    crop_size: tuple[int, int] = (256, 256)


class RAEDecoderTrainAugmentations(grain.RandomMapTransform):
    def __init__(self, cfg: RAEDecoderAugConfig, data_cfg: DataConfig):
        super().__init__()
        transforms = [
            A.Resize(*cfg.resize, interpolation=cv2.INTER_AREA),
            A.RandomCrop(*cfg.crop_size, pad_if_needed=True, pad_position="random"),
            # A.HorizontalFlip(),
            A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std),
        ]
        self.transforms = A.Compose(transforms)

    def random_map(self, element: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        seed_albumentations(rng, self.transforms)
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}


class RAEDecoderValAugmentations(grain.MapTransform):
    def __init__(self, cfg: RAEDecoderAugConfig, data_cfg: DataConfig):
        super().__init__()
        self.crop_size = cfg.crop_size[0]
        self.normalize = A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std)

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Use ADM center crop (handles large images with iterative BOX downsampling)
        image = adm_center_crop(element["image"], self.crop_size)
        # Normalize
        image = self.normalize(image=image)["image"]
        return {"image": image, "label": element["label"]}


class RAEv2GithubDecoderAugmentations(grain.MapTransform):
    """Image transform used by the released RAEv2 ImageNet stage-one loader.

    The official Arrow data were prepared by REPA's ``center-crop-dhariwal``
    transform and are already square 256px images.  Applying the same
    Albumentations-wrapped ADM transform is therefore an identity for Arrow and
    performs the required BOX/bicubic center crop for raw TFDS images.  Neither
    path uses a random crop or horizontal flip.
    """

    def __init__(self, cfg: RAEDecoderAugConfig, data_cfg: DataConfig):
        if cfg.crop_size != (256, 256):
            raise ValueError("The GitHub RAEv2 ImageNet recipe requires 256x256 images.")
        self.image_size = cfg.crop_size
        self.transforms = A.Compose(
            [
                ADMCenterCrop(cfg.crop_size[0]),
                A.Normalize(
                    data_cfg.normalization_mean,
                    data_cfg.normalization_std,
                ),
            ]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        image = self.transforms(image=element["image"])["image"]
        return {"image": image, "label": element["label"]}
