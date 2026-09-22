import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from pooldino.data import DataLoaders
from pooldino.metrics.utils import precompute_features


@dataclass
class LinearProbeConfig:
    seed: int = 42
    epochs: int = 90
    batch_size: int = 16384
    warmup_fraction: float = 0.1

    # MAE-style LARS linear probing uses a base LR of 0.1 at batch size 256,
    # scaled linearly with the optimizer batch size. For the default 16,384
    # batch this is 6.4. Sweep one and two octaves around that reference.
    learning_rates: list[float] = field(default_factory=lambda: [1.6, 3.2, 6.4, 12.8])
    weight_decays: list[float] = field(default_factory=lambda: [0.0])


class LinearClassifier(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(input_dim, output_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


def train_classifier(
    cfg: LinearProbeConfig,
    base_lr: float,
    weight_decay: float,
    mesh: jax.sharding.Mesh,
    train_feats: jax.Array,
    train_lbls: jax.Array,
    *,
    num_classes: int = 1_000,
):
    if cfg.epochs <= 0:
        raise ValueError("Linear-probe epochs must be positive.")
    if cfg.batch_size <= 0:
        raise ValueError("Linear-probe batch_size must be positive.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if not 0.0 <= cfg.warmup_fraction < 1.0:
        raise ValueError("Linear-probe warmup_fraction must be in [0, 1).")

    @nnx.jit
    def train_step(
        classifier: LinearClassifier, optim: nnx.Optimizer, feats: jax.Array, y: jax.Array
    ):
        def loss_fn(classifier: LinearClassifier, x: jax.Array, y: jax.Array):
            y_pred_logits = classifier(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(y_pred_logits, y)
            return jnp.mean(loss)

        loss, grads = nnx.value_and_grad(loss_fn)(classifier, feats, y)
        optim.update(classifier, grads)
        return loss

    samples_per_epoch = (train_feats.shape[0] // jax.device_count()) * jax.device_count()
    if samples_per_epoch == 0:
        raise ValueError("Linear probing requires at least one sample per JAX device.")
    probe_batch_size = min(cfg.batch_size, samples_per_epoch)
    total_iterations = max(
        (samples_per_epoch * cfg.epochs) // probe_batch_size,
        cfg.epochs,
    )
    classifier = LinearClassifier(train_feats.shape[-1], num_classes, rngs=nnx.Rngs(cfg.seed))

    warmup_steps = max(int(total_iterations * cfg.warmup_fraction), 1)
    lr_sched = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_iterations, warmup_steps + 1),
        end_value=0.0,
    )
    chain = optax.chain(
        optax.clip_by_global_norm(10.0), optax.lars(lr_sched, weight_decay=weight_decay)
    )
    optim = nnx.Optimizer(classifier, tx=chain, wrt=nnx.Param)

    id_rng = np.random.default_rng(cfg.seed)
    pbar = tqdm(
        range(total_iterations),
        desc="train",
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
    )
    for _ in pbar:
        ids = id_rng.choice(samples_per_epoch, size=probe_batch_size, replace=False)
        feats, y = train_feats[ids], train_lbls[ids]

        loss = train_step(classifier, optim, feats, y)
        pbar.set_postfix({"loss": loss.item()})

    return classifier


def eval_top1_classifier(
    cfg: LinearProbeConfig,
    classifier: LinearClassifier,
    val_feats: jax.Array,
    val_lbls: jax.Array,
    mesh: jax.sharding.Mesh,
):
    @nnx.jit
    def eval_step(classifier: LinearClassifier, feats: jax.Array, y: jax.Array):
        y_pred = classifier(feats)
        y_pred = jnp.argmax(y_pred, axis=1, keepdims=False)
        correct = jnp.sum(y_pred == y)
        return correct

    samples_per_epoch = (val_feats.shape[0] // jax.device_count()) * jax.device_count()
    total_iterations = (samples_per_epoch + cfg.batch_size - 1) // cfg.batch_size

    correct = 0
    total_samples = 0
    all_ids = np.arange(samples_per_epoch)
    for i in tqdm(
        range(total_iterations),
        desc="validating",
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
    ):
        start, end = i * cfg.batch_size, min((i + 1) * cfg.batch_size, samples_per_epoch)
        ids = all_ids[start:end]
        feats = jax.device_put(val_feats[ids], NamedSharding(mesh, P("data", None)))
        y = jax.device_put(val_lbls[ids], NamedSharding(mesh, P("data")))

        batch_correct = eval_step(classifier, feats, y)

        correct += batch_correct.item()
        total_samples += feats.shape[0]

    top_1 = correct / total_samples
    return top_1


def linear_probe(
    cfg: LinearProbeConfig,
    collect_batch_size: int,
    feat_fn: Callable,
    data: DataLoaders,
    *,
    mesh: jax.sharding.Mesh,
    cache_features: bool = False,
    num_classes: int = 1_000,
    output_path: Path | None = Path("top1.csv"),
):
    if cache_features:
        train_feats, train_lbls, val_feats, val_lbls = precompute_features(
            collect_batch_size, feat_fn, data, mesh=mesh
        )

        results = []
        for base_lr, wd in itertools.product(cfg.learning_rates, cfg.weight_decays):
            classifier = train_classifier(
                cfg,
                base_lr,
                wd,
                mesh,
                train_feats,
                train_lbls,
                num_classes=num_classes,
            )

            top_1 = eval_top1_classifier(cfg, classifier, val_feats, val_lbls, mesh)
            print(f"base_lr={base_lr}, wd={wd}, top_1={top_1}")
            results.append({"base_lr": base_lr, "wd": wd, "top_1": top_1})

        df = pd.DataFrame(results)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Grid search results saved to {output_path}")
        return df
    else:
        raise NotImplementedError()
