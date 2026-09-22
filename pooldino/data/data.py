import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"

import albumentations as A
import cv2
import grain.python as grain
import numpy as np
import tensorflow_datasets as tfds

from pooldino.utils import is_running_on_gcp

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def seed_albumentations(rng: np.random.Generator, *transforms: A.Compose) -> None:
    """Re-seed albumentations Compose instances with independent seeds from a grain RNG.

    Each transform receives a unique seed derived from the same RNG, ensuring
    reproducible but independent augmentations across multiple pipelines.
    """
    seeds = rng.integers(2**31, size=len(transforms))
    for t, seed in zip(transforms, seeds):
        t.set_random_seed(int(seed))


def get_interpolation(source_resolution: int, target_resolution: int) -> int:
    """Select interpolation method based on whether we're upsampling or downsampling.

    Args:
        source_resolution: Typical source image resolution
        target_resolution: Target resolution after resize

    Returns:
        OpenCV interpolation flag (cv2.INTER_LINEAR for upsampling, cv2.INTER_AREA for downsampling)
    """
    if target_resolution > source_resolution:
        return cv2.INTER_LINEAR  # Upsampling: use bilinear
    else:
        return cv2.INTER_AREA  # Downsampling: use area averaging


# Dataset presets: (tfds_name, num_classes, train_split, val_split, source_resolution)
# source_resolution is the typical image size (used to select interpolation method)
DATASET_PRESETS: dict[str, tuple[str, int, str, str, int]] = {
    "imagenet": ("imagenet2012", 1000, "train", "validation", 256),
}


def _ade20k_semantic_source(split: str, data_dir: str | None):
    from pooldino.data.ade20k import ade20k_semantic_data_source

    return ade20k_semantic_data_source(split, data_dir)


# Subset datasets that require index filtering instead of direct TFDS loading.
_SUBSET_FACTORIES: dict[str, Callable] = {
    "ade20k_semantic": _ade20k_semantic_source,
}


@dataclass
class DataConfig:
    num_workers: int = 8
    seed: int = 0

    normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD

    dataset: str = "imagenet2012"
    num_classes: int = 1000
    train_name: str = "train"
    val_name: str = "validation"
    source_resolution: int = 256
    """Typical source image resolution (used to select interpolation method)."""

    backend: Literal["tfds", "raev2_hf", "imagefolder"] = "tfds"
    """Dataset storage backend.

    ``raev2_hf`` reads the exact Hugging Face Arrow layout used by the
    released nanovisionx/RAEv2 repository. ``imagefolder`` reads class-folder
    trees such as ``<root>/train/<synset>/*.JPEG`` and
    ``<root>/val/<synset>/*.JPEG``. Keeping ``tfds`` as the default preserves
    every existing experiment and checkpoint.
    """
    data_dir: str | None = None
    """Local backend root. For ``raev2_hf`` this contains
    ``imagenet-latents-images``; for ``imagefolder`` it contains class-folder
    split directories such as ``train`` and ``val``."""

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "DataConfig":
        """Create a DataConfig from a preset name.

        Args:
            name: Preset name (e.g., 'cifar10', 'flowers102', 'pets').
                  See DATASET_PRESETS for available presets.
            **kwargs: Override any DataConfig field.

        Returns:
            DataConfig with preset values, optionally overridden.
        """
        if name not in DATASET_PRESETS:
            available = ", ".join(sorted(DATASET_PRESETS.keys()))
            raise ValueError(f"Unknown dataset preset '{name}'. Available: {available}")

        tfds_name, num_classes, train_split, val_split, source_res = DATASET_PRESETS[name]
        return cls(
            dataset=tfds_name,
            num_classes=num_classes,
            train_name=train_split,
            val_name=val_split,
            source_resolution=source_res,
            **kwargs,
        )


@dataclass(frozen=True)
class DataLoaders:
    train_loader: grain.DataLoader
    val_loader: grain.DataLoader
    train_ds_size: int
    val_ds_size: int


@dataclass(frozen=True)
class RAEv2EpochSemantics:
    """Opt-in epoch accounting used by the released RAEv2 trainers.

    RAEv2 starts a fresh distributed dataloader pass for every epoch. It drops
    any incomplete per-rank micro batch and incomplete gradient-accumulation
    group instead of carrying those records into the next epoch. Grain's
    default infinite sampler intentionally carries them, so exact-source
    experiments must request this behavior explicitly.
    """

    global_batch_size: int
    source_world_size: int = 8
    grad_accum_steps: int = 1
    shuffle_seed: int = 0


def raev2_steps_per_epoch(
    dataset_size: int,
    global_batch_size: int,
    *,
    world_size: int = 8,
    grad_accum_steps: int = 1,
) -> int:
    """Return the optimizer updates in one released-code RAEv2 epoch."""

    values = {
        "dataset_size": dataset_size,
        "global_batch_size": global_batch_size,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    denominator = world_size * grad_accum_steps
    if global_batch_size % denominator != 0:
        raise ValueError(
            "global_batch_size must be divisible by "
            "world_size * grad_accum_steps."
        )

    # DistributedSampler pads each rank to ceil(N / W), DataLoader(drop_last)
    # drops a partial per-rank micro batch, and the engine only steps on a
    # complete gradient-accumulation group.
    samples_per_rank = (dataset_size + world_size - 1) // world_size
    micro_batch_per_rank = global_batch_size // denominator
    micro_batches_per_rank = samples_per_rank // micro_batch_per_rank
    return micro_batches_per_rank // grad_accum_steps


def raev2_total_updates(
    dataset_size: int,
    global_batch_size: int,
    epochs: int,
    *,
    world_size: int = 8,
    grad_accum_steps: int = 1,
) -> int:
    """Return RAEv2's exact ``epochs * steps_per_epoch`` update count."""

    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}.")
    return epochs * raev2_steps_per_epoch(
        dataset_size,
        global_batch_size,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
    )


class EpochTruncatedIndexSampler:
    """Drop each shuffled epoch's tail before Grain batches the stream.

    ``records_per_epoch`` is the number of records consumed globally in one
    epoch. The wrapped shuffle is indexed with an ``epoch * num_records``
    stride, retaining Grain's deterministic ``seed + epoch`` permutations.
    This is a pure index mapping, so Grain's normal iterator checkpoint is
    sufficient for exact mid-epoch resume.
    """

    def __init__(
        self,
        num_records: int,
        records_per_epoch: int,
        *,
        num_epochs: int | None,
        shard_options: grain.ShardOptions = grain.NoSharding(),
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_records <= 0:
            raise ValueError("num_records must be positive.")
        if records_per_epoch <= 0:
            raise ValueError("records_per_epoch must be positive.")
        if num_epochs is not None and num_epochs <= 0:
            raise ValueError("num_epochs must be positive or None.")
        if records_per_epoch % shard_options.shard_count != 0:
            raise ValueError(
                "records_per_epoch must be divisible by the data shard count."
            )
        if seed < 0 or seed >= 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")

        self._num_records = int(num_records)
        self._records_per_epoch = int(records_per_epoch)
        self._num_epochs = num_epochs
        self._shard_options = shard_options
        self._shuffle = bool(shuffle)
        self._seed = int(seed)
        self._max_index = (
            None if num_epochs is None else self._records_per_epoch * num_epochs
        )
        # DataLoader shards the retained logical stream. Keeping this global
        # permutation unsharded matches DistributedSampler's shuffle-then-rank
        # ordering.
        self._base = grain.IndexSampler(
            num_records=self._num_records,
            shard_options=grain.NoSharding(),
            shuffle=self._shuffle,
            num_epochs=None,
            seed=self._seed,
        )

    def __len__(self) -> int:
        return sys.maxsize if self._max_index is None else self._max_index

    def __repr__(self) -> str:
        return (
            "EpochTruncatedIndexSampler("
            f"num_records={self._num_records}, "
            f"records_per_epoch={self._records_per_epoch}, "
            f"num_epochs={self._num_epochs}, "
            f"shard_options={self._shard_options!r}, "
            f"shuffle={self._shuffle}, seed={self._seed})"
        )

    def __getitem__(self, index: int) -> grain.RecordMetadata:
        if index < 0 or (self._max_index is not None and index >= self._max_index):
            raise IndexError(
                f"Sampler index {index} is outside [0, {self._max_index})."
            )
        epoch, index_in_retained_epoch = divmod(index, self._records_per_epoch)
        # DistributedSampler can pad an epoch by repeating its shuffled prefix.
        index_in_source_epoch = index_in_retained_epoch % self._num_records
        source_index = epoch * self._num_records + index_in_source_epoch
        source_metadata = self._base[source_index]
        return grain.RecordMetadata(
            index=index,
            record_key=source_metadata.record_key,
            # Padded duplicates receive independent deterministic augmentation.
            rng=np.random.Generator(np.random.Philox(key=self._seed + index)),
        )


class _RAEv2HfImageDataSource:
    """Grain-compatible view of RAEv2's preprocessed ImageNet Arrow data.

    The released loader stores training examples directly in
    ``imagenet-latents-images`` and validation examples in its ``val``
    subdirectory.  Samples are converted to the same NumPy dictionary shape
    returned by TFDS so all existing Grain transforms remain reusable.
    """

    def __init__(self, data_dir: str, split: str):
        from datasets import load_from_disk

        arrow_root = Path(data_dir).expanduser() / "imagenet-latents-images"
        is_validation = split in {"val", "validation", "test"}
        dataset_path = arrow_root / "val" if is_validation else arrow_root
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"RAEv2 Arrow split {split!r} was not found at {dataset_path}. "
                "Download nanovisionx/RAEv2-data or set DataConfig.data_dir."
            )
        self.dataset = load_from_disk(str(dataset_path))

    def __len__(self) -> int:
        return len(self.dataset)

    def labels(self, count: int | None = None) -> np.ndarray:
        """Read labels without decoding the corresponding Arrow images."""

        if count is None:
            count = len(self.dataset)
        if not 0 <= count <= len(self.dataset):
            raise ValueError(
                f"count must lie in [0, {len(self.dataset)}], got {count}."
            )
        if count == 0:
            return np.empty((0,), dtype=np.int32)
        return np.asarray(self.dataset[:count]["label"], dtype=np.int32)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample = self.dataset[int(index)]
        image = sample["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Expected an RGB image from the RAEv2 Arrow dataset, got {image.shape}."
            )
        return {
            "image": image,
            "label": np.asarray(sample["label"], dtype=np.int32),
        }


class _ImageFolderDataSource:
    """Grain-compatible deterministic view of a class-folder image tree."""

    _IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})

    def __init__(self, data_dir: str, split: str):
        split_dir_name = "val" if split in {"val", "validation", "test"} else split
        split_dir = Path(data_dir).expanduser() / split_dir_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"ImageFolder split {split!r} was not found at {split_dir}."
            )

        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not class_dirs:
            raise ValueError(f"ImageFolder split {split_dir} has no class directories.")
        self.class_names = tuple(path.name for path in class_dirs)
        self.samples = [
            (path, label)
            for label, class_dir in enumerate(class_dirs)
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in self._IMAGE_SUFFIXES
        ]
        if not self.samples:
            raise ValueError(f"ImageFolder split {split_dir} contains no images.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        image_path, label = self.samples[int(index)]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode ImageFolder image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {
            "image": np.asarray(image, dtype=np.uint8),
            "label": np.asarray(label, dtype=np.int32),
        }


def create_dataloaders(
    cfg: DataConfig,
    batch_size: int,
    train_epochs: int | None = None,
    val_epochs: int | None = None,
    train_aug: grain.MapTransform | grain.RandomMapTransform | None = None,
    val_aug: grain.MapTransform | grain.RandomMapTransform | None = None,
    drop_remainder_train: bool = True,
    drop_remainder_val: bool = False,
    val_shuffle: bool = False,
    shard_index: int | None = None,
    shard_count: int | None = None,
    gcs_bucket: str | None = None,
    train_epoch_semantics: RAEv2EpochSemantics | None = None,
    train_shuffle: bool = True,
) -> DataLoaders:
    """Create dataset loaders.

    Args:
      cfg (DataConfig)
      batch_size (int): Per-host batch size (will be sharded across hosts if shard_count > 1)
      train_epochs (int | None): None means infinite epochs.
      val_epochs (int | None): None means infinite epochs.
      train_aug (grain.MapTransform | grain.RandomMapTransform | None)
      val_aug (grain.MapTransform | grain.RandomMapTransform | None)
      drop_remainder_train (bool): Drop incomplete final batches for training
      drop_remainder_val (bool): Drop incomplete final batches for validation
      val_shuffle (bool): Whether to shuffle validation data
      shard_index (int | None): Index of current host/process for multi-host sharding.
          If None, uses jax.process_index() when jax.process_count() > 1, else no sharding.
      shard_count (int | None): Total number of hosts/processes for multi-host sharding.
          If None, uses jax.process_count() when > 1, else no sharding.
      gcs_bucket (str | None): the Google Cloud Storage bucket name where the dataset is stored.
          It assumes the following structure: gs://<gcs_bucket>/<cfg.dataset>.
          If not specified, the default tfds data_dir will be used.
      train_epoch_semantics: Optional released-RAEv2 epoch accounting. When
          set, each shuffled training epoch is truncated independently before
          batching. The default preserves the legacy continuous Grain stream.
      train_shuffle: Whether to shuffle the training split. Defaults to the
          historical behavior; source statistics explicitly disable it.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    tfds_data_dir = None
    if gcs_bucket is not None:
        if cfg.backend != "tfds":
            raise ValueError(
                "gcs_bucket is only supported by the TFDS backend; mount or copy "
                "the RAEv2 Arrow dataset locally and use DataConfig.data_dir."
            )
        if not is_running_on_gcp():
            raise RuntimeError(
                f"gcs_bucket='{gcs_bucket}' specified but not running on GCP. "
                "GCS dataset loading requires running on a GCP instance."
            )
        tfds_data_dir = f"gs://{gcs_bucket}"

    # Determine sharding options for multi-host training
    # Import jax here to avoid importing it at module level (affects JAX_PLATFORMS)
    import jax

    if shard_index is None and shard_count is None:
        # Auto-detect multi-host setup
        if jax.process_count() > 1:
            shard_index = jax.process_index()
            shard_count = jax.process_count()

    if shard_count is not None and shard_count > 1:
        shard_options = grain.ShardOptions(
            shard_index=shard_index,
            shard_count=shard_count,
            drop_remainder=True,
        )
    else:
        shard_options = grain.NoSharding()

    def create_split(
        dataset_name,
        split_name,
        aug,
        epochs,
        drop_remainder: bool,
        shuffle: bool,
        epoch_semantics: RAEv2EpochSemantics | None = None,
    ) -> tuple[grain.DataLoader, int]:
        if cfg.backend == "raev2_hf":
            if cfg.data_dir is None:
                raise ValueError(
                    "DataConfig.data_dir is required when backend='raev2_hf'."
                )
            dataset = _RAEv2HfImageDataSource(cfg.data_dir, split_name)
        elif cfg.backend == "imagefolder":
            if cfg.data_dir is None:
                raise ValueError(
                    "DataConfig.data_dir is required when backend='imagefolder'."
                )
            dataset = _ImageFolderDataSource(cfg.data_dir, split_name)
        elif cfg.backend != "tfds":
            raise ValueError(f"Unsupported dataset backend: {cfg.backend!r}.")
        elif dataset_name in _SUBSET_FACTORIES:
            subset_data_dir = (
                cfg.data_dir if cfg.data_dir is not None else tfds_data_dir
            )
            dataset = _SUBSET_FACTORIES[dataset_name](
                split_name, subset_data_dir
            )
        else:
            dataset = tfds.data_source(
                dataset_name,
                split=split_name,
                data_dir=tfds_data_dir,
            )  # ty: ignore
        operations = [grain.Batch(batch_size, drop_remainder=drop_remainder)]
        if aug is not None:
            operations.insert(0, aug)
        if epoch_semantics is None:
            sampler = grain.IndexSampler(
                num_records=len(dataset),
                num_epochs=epochs,
                shard_options=shard_options,
                shuffle=shuffle,
                seed=0,
            )
        else:
            steps_per_epoch = raev2_steps_per_epoch(
                len(dataset),
                epoch_semantics.global_batch_size,
                world_size=epoch_semantics.source_world_size,
                grad_accum_steps=epoch_semantics.grad_accum_steps,
            )
            records_per_epoch = steps_per_epoch * epoch_semantics.global_batch_size
            if records_per_epoch <= 0:
                raise ValueError(
                    "RAEv2 epoch semantics produced zero complete optimizer "
                    "updates for this dataset and global batch size."
                )
            local_records_per_epoch = records_per_epoch // shard_options.shard_count
            if local_records_per_epoch % batch_size != 0:
                raise ValueError(
                    "The retained per-shard epoch size must be divisible by "
                    "the local loader batch size. Check global batch, gradient "
                    "accumulation, and process sharding settings."
                )
            sampler = EpochTruncatedIndexSampler(
                num_records=len(dataset),
                records_per_epoch=records_per_epoch,
                num_epochs=epochs,
                shard_options=shard_options,
                shuffle=shuffle,
                seed=epoch_semantics.shuffle_seed,
            )
        loader = grain.DataLoader(
            data_source=dataset,
            operations=operations,
            sampler=sampler,
            worker_count=cfg.num_workers,
            read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=32),
        )
        return loader, len(dataset)

    t_ld, ts = create_split(
        cfg.dataset,
        cfg.train_name,
        train_aug,
        train_epochs,
        drop_remainder=drop_remainder_train,
        shuffle=train_shuffle,
        epoch_semantics=train_epoch_semantics,
    )
    v_ld, vs = create_split(
        cfg.dataset,
        cfg.val_name,
        val_aug,
        val_epochs,
        drop_remainder=drop_remainder_val,
        shuffle=val_shuffle,
    )

    return DataLoaders(
        train_loader=t_ld,
        val_loader=v_ld,
        train_ds_size=ts,
        val_ds_size=vs,
    )
