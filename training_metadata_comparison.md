# Training metadata comparison: Kohya vs ai-toolkit

Both runs train the same dataset (`Gia and Joe Combat Carl PSA`, 20 images) as an SDXL LoRA.
Lora source: https://civitai.red/models/2725726/ 
- **Kohya:** `IL_Gia_and_Joe_Combat_Carl_PSA_Kohya` (sd-scripts, commit `f931705`)
- **ai-toolkit:** `_IL__Gia_and_Joe__Combat_Carl_PSA__epoch_10` (ai-toolkit 0.10.15, diffusers)

## Settings comparison

| Setting | Kohya (sd-scripts) | ai-toolkit 0.10.15 |
|---|---|---|
| **LoRA alpha** | 16 | 32 |
| **Epochs** | 15 | 10 (metadata says `ss_epoch: 1`, notes say "Epoch 1") |
| **Total steps** | 870 | 2000 max; checkpoint at step 1800 |
| **Dataset repeats** | 10 (20 images → 200 samples/epoch) | 1 (20 images) |
| **Batch size** | 4 per device | 1 |
| **LR scheduler** | cosine_with_restarts | cosine |
| **Warmup steps** | 0 | not recorded |
| **Caption dropout** | 0 (tag dropout 0) | 0.05 |
| **Latent caching** | on | off |
| **Clip skip / encoder layer** | 1 | not recorded |
| **Multires noise** | 6 iterations, discount 0.3 | not recorded |
| **Bucketing** | on, 256–2048 (18 buckets) | not recorded |
| **Shuffle caption / keep tokens** | true / 1 | not recorded |
| **Flip / color aug / random crop** | all off | not recorded |
| **Seed** | 2423904581 | not recorded |
| **Debiased estimation / zero-terminal-SNR** | False / False | not recorded |
| **Prior loss weight** | 1 | not recorded |
| **Trainer / commit** | sd-scripts `f931705` | ai-toolkit 0.10.15, diffusers |
| **Architecture tag** | stable-diffusion-xl-v1-base/lora | stable-diffusion-xl/lora |
| **Training time** | 1h 8m 50s (17:50:40 → 18:59:30) | not recorded |
| **Tag metadata** | full booru tag frequencies | single empty caption `""` |

## Settings that match

- Network dim: 32 (both)
- UNet learning rate: 5e-4 (both)
- Text encoder learning rate: 5e-5 (both)
- Optimizer: Adafactor, scale_parameter=False (both)
- Min-SNR gamma: 5 (both)
- Noise offset: 0.1 (both)
- Mixed precision: bf16 (both)
- Gradient checkpointing: on (both)
- Gradient clipping (max grad norm): 1 (both)
- Gradient accumulation: 1 (both)
- Full fp16: off (both)
- Resolution: 1024x1024 (both)
- Base model: Civitai model version 889818 (both reference it)

## Caveats

- **ai-toolkit metadata is internally inconsistent.** It says 10 epochs with 20 batches per epoch, which is 200 steps, but `ss_max_train_steps` is 2000 and the checkpoint is labelled step 1800, epoch 30. The filename says `epoch_10`, but the internal name is `epoch_1`. It is unclear which checkpoint this file describes.
- **The ai-toolkit tag metadata is empty.** Its only entry is `"1_": {"": 1}`, so the captions used cannot be determined from the metadata. The Kohya run has full booru-style tags. Caption differences may matter as much as the settings.
- **Effective LoRA scale differs.** Alpha/dim is 0.5 for Kohya and 1.0 for ai-toolkit, so the same learning rate produces a weaker update in Kohya.
