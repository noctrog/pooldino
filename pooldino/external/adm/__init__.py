"""ADM (OpenAI guided-diffusion) evaluation suite.

This module contains the FID/sFID/IS/Precision/Recall evaluator from OpenAI's
guided-diffusion repository, which has become the standard for evaluating
diffusion models on ImageNet.

Source: https://github.com/openai/guided-diffusion
License: MIT

Note: The Evaluator class requires TensorFlow. For just downloading/loading
reference images without TensorFlow, use the functions exported here directly.
"""

from pooldino.external.adm.pack_images import center_crop_arr, create_npz_from_image_folder
from pooldino.external.adm.validation import (
    IMAGENET_256_REF_PATH,
    download_imagenet_256_ref,
    download_imagenet_ref,
    get_imagenet_256_val,
    get_imagenet_val,
)

__all__ = [
    "IMAGENET_256_REF_PATH",
    "center_crop_arr",
    "create_npz_from_image_folder",
    "download_imagenet_256_ref",
    "download_imagenet_ref",
    "get_imagenet_256_val",
    "get_imagenet_val",
]
