# AA Model Changelog

Training run history for the temporal anti-aliasing model.

## aa-v0.0.4 (2026-03-16)

**Changes:** Fixed MV scaling bug in TAA ground truth — motion vectors were ~1000× too small (normalized instead of pixel-space, missing Y-axis negation for NDC→pixel flip), making temporal accumulation a no-op and producing blurry GT. Pre-scale MVs during burst preprocessing. Also increased workers 18→20.

| Parameter | Value |
|-----------|-------|
| Dataset | 88 bursts (packed .burst) |
| Epochs | 100 (20 pretrain + 80 full) |
| Batch size | 4 |
| Patch size | 128 |
| Workers | **20** |
| Model | base_ch=16, temporal_ch=32, groups=4 |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, perc_out=0, perc_res=0, temporal=0.05, reg=0 |
| Edge boost | 5.0 |
| Device | AmdDevice(0) |
| Training time | ~96 min (5735s) |
| Log | `aa-train-v0.0.4.log` |

**Results:**

| Metric | Best | Final (ep100) |
|--------|------|---------------|
| psnr_m | 47.2 (ep98) | 44.5 |
| psnr_s | 45.9 (ep98) | 42.1 |
| gap (m-s) | +1.5 (ep78) | +2.4 |
| ssim_m | 0.977 (ep98) | 0.970 |
| charb | 0.0048 (ep84) | 0.0058 |

**Notes:**
- Best-epoch metrics (ep98: PSNR 47.2, SSIM 0.977) match v0.0.3 peak (47.0/0.977) and v0.0.2 peak (47.3/0.983).
- Final-epoch metrics lower than v0.0.3 (PSNR 44.5 vs 45.9) — high inter-epoch volatility means final ≠ best.
- PRE→FULL transition smooth: psnr_m 43.4→43.6 (no dip at all).
- **Key fix:** Previous models (v0.0.1–v0.0.3) trained against blurry TAA GT because MVs were unnormalized — temporal accumulation was effectively a no-op. This fix means v0.0.4 is the first model trained with correct multi-frame TAA ground truth.
- Despite correct GT, final metrics are similar to v0.0.3 — suggests the model was already learning spatial AA patterns, and temporal signal needs more data/epochs to show improvement.
- Workers 20 vs 18 had no measurable impact on convergence or speed (~96 vs ~95 min).

---

## aa-v0.0.3 (2026-03-15)

**Changes:** Edge-weighted Charbonnier loss (Sobel, boost=5.0), perceptual losses disabled (perc_out=0, perc_res=0).

| Parameter | Value |
|-----------|-------|
| Dataset | 88 bursts (packed .burst) |
| Epochs | 100 (20 pretrain + 80 full) |
| Batch size | 4 |
| Patch size | 128 |
| Workers | 18 |
| Model | base_ch=16, temporal_ch=32, groups=4 |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, **perc_out=0**, **perc_res=0**, temporal=0.05, reg=0 |
| **Edge boost** | **5.0** (Sobel edge weight: edges ~5x, flat ~0.2x) |
| Device | AmdDevice(0) |
| Training time | ~95 min (5720s) |
| Log | `aa-train-v0.0.3.log` |

**Results:**

| Metric | Best | Final (ep100) |
|--------|------|---------------|
| psnr_m | 47.0 (ep86) | 45.9 |
| psnr_s | 46.0 (ep86) | 44.2 |
| gap (m-s) | +2.2 (ep92) | +1.7 |
| ssim_m | 0.977 (ep94) | 0.976 |
| charb | 0.0046 (ep94) | 0.0054 |

**Notes:**
- Final metrics nearly identical to v0.0.2 (PSNR 45.9 vs 45.3, SSIM 0.976 vs 0.977).
- charb_out values not directly comparable — edge weighting inflates loss on edge pixels.
- PRE→FULL transition smooth: psnr_m 42.9→41.5 (minor dip, no collapse).
- Higher volatility than v0.0.2 (PSNR ±3 dB between epochs) — likely 88 bursts too small for stable convergence.
- Disabling perceptual losses had no negative effect — VGG feature matching adds overhead without benefit on sub-pixel AA corrections.
- Whether edge-weighted residuals are more visible at 4K requires `aa-eval` verification.

---

## aa-v0.0.2 (2026-03-15)

**Changes:** Disabled residual regularizer (reg_weight=0.0), extended pretrain from 10 to 20 epochs.

| Parameter | Value |
|-----------|-------|
| Dataset | 88 bursts (packed .burst) |
| Epochs | 100 (20 pretrain + 80 full) |
| Batch size | 4 |
| Patch size | 128 |
| Workers | 16 |
| Model | base_ch=16, temporal_ch=32, groups=4 |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, perc_out=0.5, perc_res=0.5, temporal=0.05, **reg=0** |
| Device | AmdDevice(0) |
| Training time | ~98 min (5901s) |
| Log | `aa-train-v0.0.2.log` |

**Results:**

| Metric | Best | Final (ep100) |
|--------|------|---------------|
| psnr_m | 47.3 (ep98) | 45.3 |
| psnr_s | 45.6 (ep98) | 42.6 |
| gap (m-s) | +3.2 (ep92) | +2.7 |
| ssim_m | 0.983 (ep92) | 0.977 |
| charb | 0.0039 (ep92) | 0.0045 |

**Notes:**
- Model consistently improves over input (positive gap) from ~epoch 28 onward.
- PRE→FULL transition (ep20→21) was smooth: psnr_m 42.2→42.8 (no collapse).
- Reg=0 freed the gradient entirely for reconstruction — charb converged 40% lower than v0.0.1.

---

## aa-v0.0.1 (2026-03-15)

**Changes:** Initial training run with all default parameters.

| Parameter | Value |
|-----------|-------|
| Dataset | 88 bursts (packed .burst) |
| Epochs | 100 (10 pretrain + 90 full) |
| Batch size | 4 |
| Patch size | 128 |
| Workers | 12 |
| Model | base_ch=16, temporal_ch=32, groups=4 |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, perc_out=0.5, perc_res=0.5, temporal=0.05, **reg=1** (target_mag=0.05) |
| Device | AmdDevice(0) |
| Training time | ~93 min (5571s) |
| Log | `aa-train-v0.0.1.log` |

**Results:**

| Metric | Best | Final (ep100) |
|--------|------|---------------|
| psnr_m | 41.4 (ep98) | 40.3 |
| psnr_s | 44.3 (ep98) | 42.1 |
| gap (m-s) | -0.9 (ep98) | -1.8 |
| ssim_m | 0.955 (ep98) | 0.951 |
| charb | 0.0064 (ep98) | 0.0075 |

**Notes:**
- Model output was consistently worse than input (negative gap).
- Reg loss (~0.041) was 5.5x larger than charb (~0.0075), dominating gradients.
- PRE→FULL transition (ep10→11) caused psnr_m collapse: 41.0→30.2 (-10.8 dB), recovered by ~ep50.
- Residual magnitudes were suppressed to ~0.003 — near-invisible corrections that added noise.
