"""ImageNet validation and reference image utilities for FID evaluation.

This module provides utilities to download/create ImageNet reference and validation
images for FID and rFID evaluation. Images are center-cropped using ADM preprocessing.

This is kept separate from evaluator.py to avoid TensorFlow dependency
when only loading pre-existing NPZ files.
"""

import os
from pathlib import Path

import numpy as np
import requests
from absl import logging
from tqdm.auto import tqdm

# Pre-computed VIRTUAL reference batches from OpenAI (InceptionV3 activations).
# See https://github.com/openai/guided-diffusion/tree/main/evaluations
_IMAGENET_REF_URLS = {
    256: "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz",
    512: "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz",
}

# Allow configuring the cache directory via environment variable
_CACHE_DIR = Path(os.environ.get("ADM_CACHE_DIR", Path.home() / ".cache" / "adm"))

# Legacy constants for backwards compatibility
IMAGENET_256_REF_PATH = _CACHE_DIR / "VIRTUAL_imagenet256_labeled.npz"


def download_imagenet_ref(image_size: int = 256) -> Path:
    """Download pre-computed ImageNet VIRTUAL reference batch for gFID evaluation.

    Available resolutions: 256, 512.

    Args:
        image_size: Target resolution (default 256).

    Returns:
        Path to the downloaded reference NPZ file.

    Raises:
        ValueError: If no pre-computed reference exists for the given resolution.
    """
    url = _IMAGENET_REF_URLS.get(image_size)
    if url is None:
        raise ValueError(
            f"No pre-computed VIRTUAL reference for {image_size}px. "
            f"Available: {sorted(_IMAGENET_REF_URLS.keys())}. "
            f"Use get_imagenet_val({image_size}) to generate from validation images instead."
        )
    filename = url.rsplit("/", 1)[-1].split("?")[0]
    path = _CACHE_DIR / filename
    if path.exists():
        return path
    logging.info("Downloading ImageNet %dpx VIRTUAL reference to %s...", image_size, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp_path.rename(path)
    return path


def download_imagenet_256_ref() -> Path:
    """Download ImageNet 256x256 reference. Alias for ``download_imagenet_ref(256)``."""
    return download_imagenet_ref(256)


def _create_validation_npz(dataset, image_size: int, path: Path, label: str) -> Path:
    """Center-crop all images from a dataset and save as an NPZ file.

    This is the shared implementation for ``get_imagenet_val`` and
    ``get_imagenet100_val``.

    Args:
        dataset: Indexable dataset where each element has an ``"image"`` key
            containing a uint8 numpy array.
        image_size: Target resolution for center-cropping.
        path: Destination NPZ file path.
        label: Human-readable dataset name for log messages (e.g. "ImageNet").

    Returns:
        ``path``, after writing the NPZ file.
    """
    from PIL import Image
    from pooldino.external.adm.pack_images import center_crop_arr

    num_images = len(dataset)
    logging.info("Found %d %s validation images.", num_images, label)

    samples = []
    for i in tqdm(range(num_images), desc=f"Processing {label} validation"):
        example = dataset[i]
        image = example["image"]  # numpy array (H, W, 3), uint8
        pil_image = Image.fromarray(image).convert("RGB")
        cropped = center_crop_arr(pil_image, image_size)
        samples.append(np.asarray(cropped).astype(np.uint8))

    samples = np.stack(samples)
    assert samples.shape == (num_images, image_size, image_size, 3)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, arr_0=samples)
    logging.info("Saved %d images to %s", samples.shape[0], path)

    return path


def get_imagenet_val(image_size: int = 256) -> Path:
    """Get ImageNet validation batch at the given resolution for FID evaluation.

    Returns a path to an NPZ file containing all 50k ImageNet validation
    images, center-cropped to ``image_size`` using ADM preprocessing.

    Auto-created from tfds on first use and cached to ~/.cache/adm/.

    Args:
        image_size: Target resolution for center-cropping (default 256).

    Returns:
        Path to the validation NPZ file.
    """
    path = _CACHE_DIR / f"imagenet{image_size}_val.npz"
    if path.exists():
        return path

    logging.info("Creating ImageNet %dx%d validation NPZ at %s...", image_size, image_size, path)
    logging.info("This only needs to be done once.")

    import tensorflow_datasets as tfds

    dataset = tfds.data_source("imagenet2012", split="validation")
    return _create_validation_npz(dataset, image_size, path, label="ImageNet")




def get_imagenet_256_val() -> Path:
    """Get ImageNet 256x256 validation batch. Alias for ``get_imagenet_val(256)``."""
    return get_imagenet_val(256)
