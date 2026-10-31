# MOSS-Image training settings

Settings used by the two MOSS-Image LoRA trainer configs. Both train with [kohya-ss sd-scripts](https://github.com/kohya-ss/sd-scripts) (>= v0.12.1), following its [MOSS-Image LoRA training guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md). They differ in the speed flags and in their model registration (see Summary of differences).

- **Base:** control run
- **Fast:** speed variant, same weights and script but different settings

Values are the module-level constants. Per-job values from the queue (`dim`, `alpha`, `learning_rate`, `epochs`, `resolution`, `repeats`, `caption_dropout`) override the defaults when set.

## Summary of differences

The training settings that differ between the two configs are:

- `GRADIENT_CHECKPOINTING`: `True` (Base) vs `False` (Fast)
- `CONVROT_INT8`: `False` vs `True`
- `COMPILE`, `COMPILE_CACHE_SIZE_LIMIT`, `CUDA_ALLOW_TF32`, `CUDA_CUDNN_BENCHMARK`: unset in Base, enabled in Fast

Other differences:

| Item | Base | Fast |
|---|---|---|
| TOML keys `compile`, `compile_cache_size_limit`, `cuda_allow_tf32`, `cuda_cudnn_benchmark` | not written | written |

The configs were tuned for an RTX 3090 (24 GB), with bf16 DiT and cached text-encoder outputs.


## Network

| Setting | Base | Fast | Notes |
|---|---|---|---|
| `network_module` | `networks.lora_generic` | same | Hard-coded |
| `DEFAULT_NETWORK_DIM` | 32 | 32 | Overridden by job `dim` |
| `DEFAULT_NETWORK_ALPHA` | 32 | 32 | Alpha = dim, so scale is 1.0 |
| `network_train_unet_only` | `True` | `True` | Hard-coded. Required by text-encoder output caching; the text encoder stays frozen |

## Optimizer and scheduler

| Setting | Base | Fast | Notes |
|---|---|---|---|
| `DEFAULT_LEARNING_RATE` | 1e-4 | 1e-4 |  |
| `OPTIMIZER_TYPE` | `AdamW8bit` | same | |
| `OPTIMIZER_WEIGHT_DECAY` | 0.1 | 0.1 | Passed as `weight_decay=0.1` in `optimizer_args` |
| `OPTIMIZER_BETAS` | [0.9, 0.99] | same | Passed as `betas=[0.9, 0.99]` in `optimizer_args` |
| `LR_SCHEDULER` | `cosine_with_restarts` | same | |
| `LR_SCHEDULER_NUM_CYCLES` | 1 | 1 | |
| `LR_WARMUP_STEPS` | 100 | 100 | |
| `MAX_GRAD_NORM` | 1.0 | 1.0 | |

## Timestep sampling (MOSS-specific)

| Setting | Base | Fast | Notes |
|---|---|---|---|
| `TIMESTEP_SAMPLING` | `sigma` | same | |
| `WEIGHTING_SCHEME` | `logit_normal` | same | |
| `LOGIT_MEAN` | -1.09 | same | Reproduces the pre-training timestep distribution ([upstream guide §7.2](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md#72-training-settings--学習設定)) |
| `LOGIT_STD` | 0.95 | same | |
| `DISCRETE_FLOW_SHIFT` | 1.0 | same | |

The config comments name a simpler near-equivalent: `TIMESTEP_SAMPLING='shift'` with `DISCRETE_FLOW_SHIFT=3.0`, dropping the `WEIGHTING_SCHEME` and `LOGIT_*` settings. Neither config uses it.

The [upstream guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md#72-training-settings--学習設定) (section 7.2) says the official schedule also truncates the distribution at 0.95 and mixes in 1% uniform samples. The sd-scripts trainer does not reproduce either, so these settings only approximate the pre-training distribution.

## Precision, memory and speed

| Setting | Base | Fast | Notes |
|---|---|---|---|
| `MIXED_PRECISION` | `bf16` | same | |
| `GRADIENT_CHECKPOINTING` | `True` | `False` | Fast skips activation recompute, so it may OOM. If a job OOMs, set it back to `True` first. The upstream guide calls it practically required for MOSS-Image |
| `CACHE_LATENTS` | `True` | same | |
| `CACHE_TEXT_ENCODER_OUTPUTS` | `True` | same | |
| `CONVROT_INT8` | `False` | `True` | INT8 DiT base weights (real INT8 GEMM). Needs triton; first ~20 steps are slow while it autotunes. Frees about 5 GB of weight VRAM |
| `CONVROT_INT8_TEXT_ENCODER` | `False` | `False` | Text encoder stays bf16 in both |
| `COMPILE` | not set (off) | `True` | Per-block `torch.compile`. ConvRot INT8 Linears are excluded from compilation, so the gain is smaller when both are on. Each new bucket shape recompiles, and the first epoch is slower |
| `COMPILE_CACHE_SIZE_LIMIT` | not set | 32 | Only used when `COMPILE` is on |
| `CUDA_ALLOW_TF32` | not set (off) | `True` | Only in Fast |
| `CUDA_CUDNN_BENCHMARK` | not set (off) | `True` | Only in Fast |
| `NUM_CPU_THREADS_PER_PROCESS` | 1 | 1 | |


## Training hyperparameters

| Setting | Base | Fast | Notes |
|---|---|---|---|
| `DEFAULT_MAX_TRAIN_EPOCHS` | 10 | 10 |  |
| `TRAIN_BATCH_SIZE` | 1 | 1 | |
| `GRADIENT_ACCUMULATION_STEPS` | 1 | 1 | |
| `SEED` | 42 | 42 | |
| `SHUFFLE_CAPTION` | `False` | `False` | |

## Dataset

| Setting | Base | Fast | Notes |
|---|---|---|---|
| `DEFAULT_RESOLUTION` | 768 | 768 |  |
| `DEFAULT_REPEATS` | 10 | 10 |  |
| `DEFAULT_CAPTION_DROPOUT` | 0.1 | 0.1 |  |
| `ENABLE_BUCKET` | `True` | same | |
| `BUCKET_NO_UPSCALE` | `False` | same | |
| `BUCKET_RESO_STEPS` | 64 | 64 |  |
| `MIN_BUCKET_RESO` | 256 | 256 | |
| `MAX_BUCKET_RESO` | 4096 | 4096 | |
| `CAPTION_EXTENSION` | `.txt` | same | |

