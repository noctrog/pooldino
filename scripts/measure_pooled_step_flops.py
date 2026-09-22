#!/usr/bin/env python3
"""Measure compiler-estimated RAEv2 pooled-generator training FLOPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chex
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax

from pooldino.models.transformer import set_attn_implementation
from pooldino.pooled_generator import (
    encode_pooled_decoder_targets_impl,
    make_pooled_generator,
    restore_pooled_decoder_components,
)
from pooldino.train_generator import (
    _apply_transport_class_dropout,
    _cosine_distance,
    _load_latent_stats,
    _raev2_training_path,
    _sample_time,
    build_optimizer_transform,
    get_experiment,
)


STATIC_NAMES = (
    "kappa",
    "xpred_denom_eps",
    "conditioning_dropout_prob",
    "num_classes",
    "backbone_resolution",
    "num_prefix_tokens",
    "layer_indices",
    "aggregation",
    "normalize_each",
    "add_final_mean",
    "post_pool_norm",
    "representation_eps",
    "grid_hw",
    "pool_hw",
    "time_mu",
    "time_sigma",
    "base_model_coeff",
    "use_self_repa",
    "self_repa_coeff",
    "self_repa_cosine_weight",
)


@nnx.jit(static_argnames=STATIC_NAMES)
def loss_and_grad(
    generator,
    dino,
    tokenizer,
    images,
    labels,
    key,
    *,
    kappa,
    xpred_denom_eps,
    conditioning_dropout_prob,
    num_classes,
    backbone_resolution,
    num_prefix_tokens,
    layer_indices,
    aggregation,
    normalize_each,
    add_final_mean,
    post_pool_norm,
    representation_eps,
    grid_hw,
    pool_hw,
    time_mu,
    time_sigma,
    base_model_coeff,
    use_self_repa,
    self_repa_coeff,
    self_repa_cosine_weight,
):
    chex.assert_rank(images, 4)
    latents, self_repa_target = encode_pooled_decoder_targets_impl(
        images,
        dino,
        tokenizer,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        post_pool_norm=post_pool_norm,
        representation_eps=representation_eps,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
    )
    latents = generator.normalize(latents)
    key_noise, key_time, key_dropout = jax.random.split(key, 3)
    noise = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    model_t = _sample_time(
        key_time,
        latents.shape[0],
        latents.dtype,
        mu=time_mu,
        sigma=time_sigma,
        kappa=kappa,
    )
    xt, target_velocity, transport_denom = _raev2_training_path(
        latents, noise, model_t, xpred_denom_eps
    )
    model_labels, _ = _apply_transport_class_dropout(
        labels,
        key_dropout,
        conditioning_dropout_prob,
        num_classes,
    )

    def loss_fn(model):
        prediction = model(
            xt,
            model_t,
            model_labels,
            train=True,
            return_base_model=True,
            return_self_repa=use_self_repa,
        )
        predicted_velocity = (xt - prediction["x"]) / transport_denom
        flow_loss = jnp.mean(jnp.square(predicted_velocity - target_velocity))
        base_velocity = (xt - prediction["base_x"]) / transport_denom
        base_loss = jnp.mean(jnp.square(base_velocity - target_velocity))
        self_repa_loss = jnp.asarray(0.0, dtype=flow_loss.dtype)
        self_repa_cosine = jnp.asarray(0.0, dtype=flow_loss.dtype)
        if use_self_repa:
            self_repa_prediction = prediction["self_repa"]
            self_repa_loss = jnp.mean(
                jnp.square(self_repa_prediction - self_repa_target)
            )
            self_repa_cosine = jnp.mean(
                _cosine_distance(self_repa_prediction, self_repa_target)
            )
        return flow_loss + base_model_coeff * base_loss + self_repa_coeff * (
            self_repa_loss + self_repa_cosine_weight * self_repa_cosine
        )

    loss, grads = nnx.value_and_grad(loss_fn)(generator)
    return loss, grads


@nnx.jit(
    static_argnames=(
        "backbone_resolution",
        "num_prefix_tokens",
        "layer_indices",
        "aggregation",
        "normalize_each",
        "add_final_mean",
        "post_pool_norm",
        "representation_eps",
        "grid_hw",
        "pool_hw",
    )
)
def encode_only(
    dino,
    tokenizer,
    images,
    *,
    backbone_resolution,
    num_prefix_tokens,
    layer_indices,
    aggregation,
    normalize_each,
    add_final_mean,
    post_pool_norm,
    representation_eps,
    grid_hw,
    pool_hw,
):
    return encode_pooled_decoder_targets_impl(
        images,
        dino,
        tokenizer,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        post_pool_norm=post_pool_norm,
        representation_eps=representation_eps,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
    )


def flops(lowered) -> float:
    analysis = lowered.cost_analysis()
    if analysis is None:
        analysis = lowered.compile().cost_analysis()
    if analysis is None:
        raise RuntimeError("JAX cost analysis returned no analysis table.")
    value = analysis.get("flops")
    if value is None:
        raise RuntimeError("JAX cost analysis did not report FLOPs.")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-path", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    args = parser.parse_args()

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )
    restored = restore_pooled_decoder_components(
        args.decoder_path,
        mesh=mesh,
        mp=mp,
        seed=42,
        step=40032,
        use_ema=True,
        restore_decoder=False,
        implementation="xla",
        source_encoder_fp32=True,
    )
    cfg = get_experiment(args.experiment, restored)
    stats_path = args.decoder_path / "pooled_latent_stats.npz"
    latent_mean, latent_std, _ = _load_latent_stats(
        stats_path,
        cfg.latent_norm_eps,
        restored=restored,
    )
    generator = make_pooled_generator(
        cfg,
        restored.latent_spec,
        mp,
        nnx.Rngs(42),
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    set_attn_implementation(generator, "xla")
    parameter_count = sum(
        value.size for value in jax.tree.leaves(nnx.state(generator, nnx.Param))
    )
    kappa = cfg.kappa
    if kappa is None and cfg.time_dist_shift_base is not None:
        shift_dim = cfg.time_dist_shift_dim
        if shift_dim is None:
            shift_dim = restored.latent_spec.num_latents * restored.latent_spec.feat
        kappa = max(1.0, (shift_dim / cfg.time_dist_shift_base) ** 0.5)

    images = jnp.zeros(
        (args.batch_size, 256, 256, 3),
        dtype=jnp.float32,
    )
    labels = jnp.zeros((args.batch_size,), dtype=jnp.int32)
    common = dict(
        backbone_resolution=restored.cfg.backbone_resolution,
        num_prefix_tokens=restored.cfg.num_prefix_tokens,
        layer_indices=restored.layer_indices,
        aggregation=restored.cfg.representation.aggregation,
        normalize_each=restored.cfg.representation.normalize_each,
        add_final_mean=restored.cfg.representation.add_final_mean,
        post_pool_norm=restored.cfg.post_pool_norm,
        representation_eps=restored.cfg.representation.eps,
        grid_hw=restored.grid_hw,
        pool_hw=restored.cfg.pool_window,
    )
    encode_lowered = encode_only.lower(
        restored.dino,
        restored.tokenizer,
        images,
        **common,
    )
    total_lowered = loss_and_grad.lower(
        generator,
        restored.dino,
        restored.tokenizer,
        images,
        labels,
        jax.random.PRNGKey(0),
        kappa=kappa,
        xpred_denom_eps=cfg.xpred_denom_eps,
        conditioning_dropout_prob=cfg.conditioning_dropout_prob,
        num_classes=cfg.raev2_ddt.num_classes,
        time_mu=cfg.time_mu,
        time_sigma=cfg.time_sigma,
        base_model_coeff=cfg.base_model_coeff,
        use_self_repa=cfg.self_repa,
        self_repa_coeff=cfg.self_repa_coeff,
        self_repa_cosine_weight=cfg.self_repa_cosine_weight,
        **common,
    )
    encode_flops = flops(encode_lowered)
    loss_grad_flops = flops(total_lowered)
    generator_flops = loss_grad_flops - encode_flops

    # Measure the exact GMuon/AdamW update and EMA once per global batch.
    params = nnx.state(generator, nnx.Param)
    grads = jax.tree.map(jnp.zeros_like, params)
    transform = build_optimizer_transform(cfg, 2e-4)
    opt_state = transform.init(params)

    @jax.jit
    def optimizer_step(current_params, current_state, current_grads):
        updates, new_state = transform.update(
            current_grads,
            current_state,
            current_params,
        )
        return optax.apply_updates(current_params, updates), new_state

    optimizer_flops = flops(optimizer_step.lower(params, opt_state, grads))

    @jax.jit
    def ema_step(target, source):
        return jax.tree.map(
            lambda old, new: old * cfg.train.ema + new * (1.0 - cfg.train.ema),
            target,
            source,
        )

    ema_flops = flops(ema_step.lower(params, params))
    per_image_flops = loss_grad_flops / args.batch_size
    per_update_flops = (
        per_image_flops * args.global_batch_size + optimizer_flops + ema_flops
    )
    result = {
        "decoder": args.decoder_path.name,
        "experiment": args.experiment,
        "device_count": jax.device_count(),
        "trace_batch_size": args.batch_size,
        "global_batch_size": args.global_batch_size,
        "latent_grid": list(restored.latent_spec.grid),
        "latent_tokens": restored.latent_spec.num_latents,
        "parameters": parameter_count,
        "kappa": kappa,
        "encode_flops_per_image": encode_flops / args.batch_size,
        "generator_forward_backward_flops_per_image": generator_flops
        / args.batch_size,
        "loss_and_grad_flops_per_image": per_image_flops,
        "optimizer_flops_per_global_update": optimizer_flops,
        "ema_flops_per_global_update": ema_flops,
        "estimated_total_flops_per_global_update": per_update_flops,
    }
    print("FLOP_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
