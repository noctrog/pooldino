# PoolDINO

**Pooling Representation Autoencoders for Efficient Diffusion**

Ramón Calvo-González · Youssef Saied · François Fleuret

[Paper](https://arxiv.org/abs/2610.09242) · [Project page](https://pooldino.ramoncalvo.com) · [Hugging Face models](https://huggingface.co/noctrog/pooldino)

Implementation of PoolDINO, a learned spatial pooling method for representation autoencoders. The package contains RGB decoder and generator training, sampling, classification probes, dense prediction, and pooling-operator analysis.

## Installation

Python 3.13 and Linux with CUDA 12 are the intended training environment. Install
[uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --locked --extra cuda --extra metrics
```

For CPU-only development, use `uv sync --locked`. The `metrics` extra includes TensorFlow on Linux for ADM evaluation and `torch-fidelity` for reconstruction FID. PyTorch uses CPU wheels; model training and sampling use JAX on the GPU. The optional `raev2_fd` metric backend additionally requires the `fd_evaluator` package distributed with the public RAEv2 evaluation setup; it is not vendored. Sampling can run with `--generate-only` before metrics are computed separately. W&B logging is **off by default**. If enabled with `--use-wandb`, configure your own account through `WANDB_ENTITY` and `WANDB_API_KEY`. No account is embedded.

## Data and pretrained weights

No images, trained checkpoints, private paths, or experiment logs are bundled. Pretrained PoolDINO checkpoints and download instructions are available on [Hugging Face](https://huggingface.co/noctrog/pooldino).

- Prepare ImageNet-1K as TFDS `imagenet2012`, and set `TFDS_DATA_DIR` to its root.
  Classification probes can also use ImageFolder `train/` and `val/` splits.
- ADE20K uses the official `ADEChallengeData2016` semantic annotations, supplied
  with `--ade20k-root`.
- The NYUv2 preparation helper is `scripts/prepare_nyuv2_tfds_from_hf.py`.
  Use its `--help` for the download and TFDS preparation options.
- Frozen DINOv3-L/16 and DINO-S/8 discriminator weights are resolved from the
  pinned public RAEv2 model release, with SHA-256 verification. To supply local
  weights, set `DINOV3_VITL16_WEIGHTS` and `RAEV2_DINO_S8_WEIGHTS`.
- Set `POOLDINO_CACHE_DIR` to relocate the model cache; its default is
  `~/.cache/pooldino`.

Obtain datasets and pretrained weights under their respective terms. Model
weights may be downloaded on the first training or evaluation invocation.

## Two-stage training

The experiment identifiers below retain their checkpoint-compatible spelling:
`repeatconv` means learned spatial pooling followed by repetition, and
`repeatpool` means fixed average pooling followed by repetition.

| Representation | Decoder experiment |
|---|---|
| Unpooled | `pool1x1-dinol-vitxl-raev2official-tfds` |
| Learned 2×2 | `repeatconv2x2-dinol-vitxl-raev2official-tfds` |
| Learned 2×4 | `repeatconv2x4-dinol-vitxl-raev2official-tfds` |
| Learned 4×2 | `repeatconv4x2-dinol-vitxl-raev2official-tfds` |
| Learned 4×4 | `repeatconv4x4-dinol-vitxl-raev2official-tfds` |
| Average 2×2 | `repeatpool2x2-dinol-vitxl-raev2official-tfds` |
| Average 4×4 | `repeatpool4x4-dinol-vitxl-raev2official-tfds` |

```bash
# Stage 1: frozen encoder, trainable pooling and RGB decoder.
uv run python -m pooldino.train_decoder \
  --experiment repeatconv2x2-dinol-vitxl-raev2official-tfds

# Compute source-bound latent normalization statistics from training data.
uv run python -m pooldino.compute_stats \
  --checkpoint output/decoders/repeatconv2x2-dinol-vitxl-raev2official-tfds

# Stage 2: freeze stage 1 and train the flow generator.
uv run python -m pooldino.train_generator \
  --pooled-decoder-path output/decoders/repeatconv2x2-dinol-vitxl-raev2official-tfds \
  --experiment raev2ddt-autokappa
```

The decoder recipe is 16 epochs; the generator defaults to 80 epochs. Use
`--epochs 180` for the extended 2×2 run or `--epochs 300` for 4×4. This changes the horizon, not the learning-rate decay endpoint. Use `--gpu-batch-size` to choose a microbatch that fits your hardware; the configured global batch is maintained by accumulation. Check `--help` for distributed/FSDP settings.

Default outputs are `output/decoders/<experiment>` and `output/generators/<decoder>/<experiment>`. Both training entry points support `--checkpoint-dir` and `--restore`. No scheduler-specific launchers are required.

## Sampling and guidance

```bash
uv run python scripts/export_raev2_condition_labels.py --help
uv run python -m pooldino.eval.gfid_pooled_decoder_adm \
  --generator-path output/generators/DECODER/GENERATOR \
  --pooled-decoder-path output/decoders/DECODER \
  --condition-labels-path /path/to/imagenet2012-validation-labels-tfds.npz \
  --protocol raev2_ig --ig-scale 2.0 --generate-only
```

`2.0` is an example scale, not a universal optimum. Select scales per checkpoint as in the paper. This named profile uses 100 Euler steps, seed 42, 50,000 images, EMA weights, and BF16 inference. Guidance uses noise-to-data time: IG is active on `[0, 0.9]`; explicit CFG uses `[0.3, 1]`. IG and CFG are neutral at scale 1. Add `--cfg-scale VALUE` to combine CFG with IG. For an unguided run, pass `--ig-scale 1 --cfg-scale 1`. The legacy encoder-reconstruction guidance remains
available for appendix ablations; do not confuse its scale convention with IG.

Time shifting is obtained from the generator configuration. Do not substitute a different shift or number of steps when comparing quality and throughput. Named profiles resolve their own settings and can override CLI defaults; inspect the saved `generation_config.txt` for the effective protocol.

Compute ADM FID, spatial FID, and Inception Score on generated uint8 NPZ samples:

```bash
uv run python -m pooldino.external.adm.evaluator /path/to/samples.npz
```

For reconstruction, use `pooldino.eval.rfid_pooled_decoder_adm`; the optional `--run-adm-evaluator` runs the selected metric backend in the current environment. Use `pooldino.eval.samples_metrics` for paired PSNR and LPIPS. Supply the matching reference NPZ explicitly: image order and preprocessing must match.

## Other paper experiments

All entry points accept `--help`.

| Experiment | Module or script |
|---|---|
| Frozen k-NN | `pooldino.eval.pooled_knn` |
| Frozen linear probe | `pooldino.eval.pooled_linear_probe` |
| Average/PCA/random probes | `pooldino.eval.pooled_baseline_probe` |
| Seeded random projection | `pooldino.eval.generate_pooled_random_projection` |
| Kernel/subspace analysis | `pooldino.eval.repeatconv_operator_analysis` |
| Train-fit/held-out operator analysis | `pooldino.eval.repeatconv_data_analysis` |
| ADE20K | `pooldino.train_segmentation` |
| NYUv2 | `pooldino.train_depth` |
| Component FLOPs | `scripts/measure_pooled_step_flops.py` |

Dense-task training defaults to a fresh task decoder with frozen encoder and
pooling. Supply `--vit-init-checkpoint` to initialize from an RGB decoder. Outputs
go to `output/segmentation` and `output/depth`. Classification uses deterministic
features, mean reduction, and L2 normalization. Generate random projections with
explicit seeds and record all draws when reporting mean and standard deviation.

## Relocating existing checkpoints

Always pass the new decoder/generator/statistics locations explicitly. Artifact checks retain the source step, EMA selection, configuration hash, and statistics hash. If an old checkpoint records a different directory layout, set an explicit prefix mapping rather than disabling these checks:

```bash
export POOLDINO_ARTIFACT_PATH_MAP='{"/old/output/decoder":"/new/output/decoders"}'
```

Use the actual absolute paths in your local environment. This remaps identity comparisons only: it neither rewrites checkpoint bytes nor changes their hashes. Keep the statistics file with its matching decoder. Sanitize metadata separately before distributing checkpoints; **this source release contains no checkpoints**.

## Tests and release scope

```bash
uv run pytest -q
uv run ruff check --select F821,F822,F823 pooldino tests scripts
```

Tests use small models and synthetic data; they do not reproduce full training runs.

## Citation

```bibtex
@misc{calvogonzález2026poolingrepresentationautoencodersefficient,
      title={Pooling Representation Autoencoders for Efficient Diffusion},
      author={Ramón Calvo-González and Youssef Saied and François Fleuret},
      year={2026},
      eprint={2610.09242},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.09242},
}
```
