"""Pack images into NPZ format for FID evaluation.

Modified from https://github.com/bytetriper/RAE/blob/main/pack_images.py
which is based on https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py

This script applies ADM-style center cropping to images and packs them into
a single NPZ file suitable for FID evaluation.

Usage:
    python -m pooldino.external.adm.pack_images /path/to/imagenet/val 256 /path/to/output
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center cropping implementation from ADM.

    Reference:
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    if pil_image.size == (image_size, image_size):
        return pil_image

    # Downsample by 2 while image is >= 2x target size (faster than single large resize)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    # Resize so shortest side equals target size
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    # Center crop to target size
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def create_npz_from_image_folder(
    image_dir: str | Path,
    image_size: int = 256,
    num: int | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Build a single .npz file from a folder of images.

    Args:
        image_dir: Directory containing images (searched recursively).
        image_size: Target image size (default 256).
        num: Maximum number of images to include (default: all).
        output_path: Output NPZ file path. If None, saves to {image_dir}.npz.

    Returns:
        Path to the created NPZ file.
    """
    image_dir = Path(image_dir)

    # Get all images under image_dir (recursively)
    img_suffix = (".png", ".jpg", ".jpeg", ".JPEG", ".PNG", ".JPG")
    imgs = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(img_suffix):
                imgs.append(os.path.join(root, file))

    # Sort for reproducibility
    imgs.sort()
    print(f"Found {len(imgs)} valid images in {image_dir}.")

    if num is not None:
        num = min(num, len(imgs))
    else:
        num = len(imgs)

    # Process images
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from images"):
        img_path = imgs[i]
        sample_pil = Image.open(img_path).convert("RGB")
        sample_pil = center_crop_arr(sample_pil, image_size)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)

    samples = np.stack(samples)
    assert samples.shape == (num, image_size, image_size, 3)

    # Determine output path
    if output_path is None:
        npz_path = image_dir.with_suffix(".npz")
    else:
        npz_path = Path(output_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python -m pooldino.external.adm.pack_images <image_dir> [image_size] [output_path]")
        sys.exit(1)

    image_dir = sys.argv[1]
    image_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 256
    output_path = sys.argv[3] if len(sys.argv) >= 4 else None

    if not os.path.isdir(image_dir):
        print(f"Invalid directory: {image_dir}")
        sys.exit(1)

    print(f"Creating .npz file from images in {image_dir}, image_size={image_size}")
    create_npz_from_image_folder(image_dir, image_size=image_size, output_path=output_path)


if __name__ == "__main__":
    main()
