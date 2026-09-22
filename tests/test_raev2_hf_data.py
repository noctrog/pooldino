from __future__ import annotations

import grain.python as grain
import numpy as np
from PIL import Image

from pooldino.data.data import (
    DataConfig,
    EpochTruncatedIndexSampler,
    RAEv2EpochSemantics,
    _RAEv2HfImageDataSource,
    create_dataloaders,
    raev2_steps_per_epoch,
    raev2_total_updates,
)


def test_raev2_hf_backend_is_opt_in():
    cfg = DataConfig()
    assert cfg.backend == "tfds"
    assert cfg.data_dir is None


def test_raev2_hf_source_uses_released_split_layout(tmp_path, monkeypatch):
    arrow_root = tmp_path / "imagenet-latents-images"
    validation_root = arrow_root / "val"
    validation_root.mkdir(parents=True)

    image = Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8), mode="RGB")

    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {"image": image, "label": 17}

    loaded = []

    def fake_load_from_disk(path):
        loaded.append(path)
        return FakeDataset()

    monkeypatch.setattr("datasets.load_from_disk", fake_load_from_disk)

    source = _RAEv2HfImageDataSource(str(tmp_path), "validation")
    sample = source[0]

    assert loaded == [str(validation_root)]
    assert len(source) == 1
    assert sample["image"].shape == (4, 5, 3)
    assert sample["image"].dtype == np.uint8
    assert sample["label"].dtype == np.int32
    assert int(sample["label"]) == 17


def _epoch_truncated_loader(*, shuffle: bool = False):
    sampler = EpochTruncatedIndexSampler(
        num_records=10,
        records_per_epoch=8,
        num_epochs=2,
        shuffle=shuffle,
        seed=0,
    )
    return grain.DataLoader(
        data_source=list(range(10)),
        operations=[grain.Batch(4, drop_remainder=True)],
        sampler=sampler,
        worker_count=0,
    )


def test_epoch_truncated_sampler_never_batches_across_epoch_boundary():
    batches = [batch.tolist() for batch in _epoch_truncated_loader()]
    assert batches == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_epoch_truncated_sampler_uses_a_new_seeded_permutation_each_epoch():
    batches = [batch.tolist() for batch in _epoch_truncated_loader(shuffle=True)]
    first_epoch = batches[0] + batches[1]
    second_epoch = batches[2] + batches[3]
    assert len(set(first_epoch)) == 8
    assert len(set(second_epoch)) == 8
    assert first_epoch != second_epoch


def test_epoch_truncated_loader_state_resumes_mid_epoch():
    loader = _epoch_truncated_loader(shuffle=True)
    iterator = iter(loader)
    first = next(iterator)
    state = iterator.get_state()
    expected_next = next(iterator)

    restored = iter(loader)
    restored.set_state(state)
    np.testing.assert_array_equal(next(restored), expected_next)
    assert first.shape == (4,)


def test_raev2_update_accounting_drops_every_epoch_remainder():
    assert raev2_steps_per_epoch(10, 4, world_size=1) == 2
    assert raev2_total_updates(10, 4, 3, world_size=1) == 6

    # Released ImageNet launch: eight ranks, global batch 1024, 80 epochs.
    assert raev2_steps_per_epoch(1_281_167, 1024, world_size=8) == 1_251
    assert raev2_total_updates(1_281_167, 1024, 80, world_size=8) == 100_080


def test_create_dataloaders_exact_epoch_option_is_narrowly_opt_in(monkeypatch):
    source = [
        {"image": np.asarray([index], dtype=np.int32), "label": index}
        for index in range(10)
    ]
    monkeypatch.setattr("tensorflow_datasets.data_source", lambda *a, **k: source)
    cfg = DataConfig(num_workers=0, dataset="fake", train_name="train", val_name="val")

    exact = create_dataloaders(
        cfg,
        batch_size=2,
        train_epochs=2,
        val_epochs=1,
        train_epoch_semantics=RAEv2EpochSemantics(
            global_batch_size=4,
            source_world_size=1,
            grad_accum_steps=2,
        ),
    )
    exact_batches = [batch["label"].tolist() for batch in exact.train_loader]
    assert len(exact_batches) == 8  # 4 micro-batches/epoch, for 2 epochs.

    legacy = create_dataloaders(
        cfg,
        batch_size=4,
        train_epochs=2,
        val_epochs=1,
    )
    legacy_batches = [batch["label"].tolist() for batch in legacy.train_loader]
    assert len(legacy_batches) == 5  # Legacy Grain carries the epoch tail.


def test_train_shuffle_can_be_disabled_for_source_statistics(monkeypatch):
    source = [
        {"image": np.asarray([index], dtype=np.int32), "label": index}
        for index in range(10)
    ]
    monkeypatch.setattr("tensorflow_datasets.data_source", lambda *a, **k: source)
    cfg = DataConfig(num_workers=0, dataset="fake", train_name="train", val_name="val")
    loaders = create_dataloaders(
        cfg,
        batch_size=10,
        train_epochs=1,
        val_epochs=1,
        train_shuffle=False,
    )
    labels = next(iter(loaders.train_loader))["label"]
    np.testing.assert_array_equal(labels, np.arange(10))
