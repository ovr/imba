# AA Model Changelog

Training run history for the temporal anti-aliasing model.

## aa-v0.0.9 (2026-04-08)

**Changes:** First cloud training run (CUDA). Same architecture and hyperparameters as v0.0.8 — validates reproducibility across AMD→CUDA. Workers reduced from 40 to 27 (cloud instance).

| Parameter | Value |
|-----------|-------|
| Dataset | 1023 bursts (packed .burst) |
| Epochs | 100 (5 pretrain + 95 full) |
| Batch size | 4 |
| Patch size | 128 |
| Workers | **27** |
| Model | base_ch=16, temporal_ch=32, groups=4, 6ch input |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, perc_out=0, perc_res=0, temporal=0.05, reg=0 |
| Edge boost | 5.0 |
| Device | **Cuda(0)** (cloud, was AmdDevice(0)) |
| Training time | ~388 min (23261s) |
| Log | `train-aa-v0.0.9.log` |

**Results:**

| Metric | Best | Final (ep100) |
|--------|------|---------------|
| psnr_m | 45.1 (ep75) | 44.8 |
| psnr_s | 43.4 (ep75) | 42.6 |
| gap (m-s) | +2.5 (ep99) | +2.2 |
| ssim_m | 0.963 (ep100) | 0.963 |
| charb | 0.00716 (ep80) | 0.00752 |

**Notes:**
- **Matches v0.0.8** — best_loss 0.00738 vs 0.00715 (~3% difference, within run-to-run variance). Confirms CUDA↔AMD reproducibility for this architecture.
- **Still behind v0.0.5/v0.0.6** (8ch architecture) — PSNR 45.1 vs 46.2–46.3, SSIM 0.963 vs 0.970–0.979, charb 0.00716 vs 0.0051–0.0062. Jitter-in-MV architecture has not yet recovered the quality gap.
- **PRE→FULL transition smooth** — psnr_m 42.7 (ep5) → 42.5 (ep6), no collapse.
- **Convergence plateaued ~ep50–60** — metrics oscillated in narrow band (PSNR 43.5–45.1, charb 0.0072–0.0082) for last 40 epochs. LR decay would likely help push past plateau.
- **Gap slowly improving** late in training — ep90–100 avg ~+2.2 vs ep20–30 avg ~+1.5, suggesting temporal fusion is learning but slowly.
- **Consistent batch time ~520ms on CUDA** — no thermal throttling (unlike v0.0.6 AMD 510→780ms).

---

## aa-v0.0.8 (2026-04-07)

**Changes:** Architecture update — removed jitter as separate input channels, now encoded into motion vectors as `mv_compensated = mv + jitter_delta`. Model input reduced from 8ch to 6ch per frame. Also reduced pretrain from 20 to 5 epochs, doubled workers from 20 to 40.

**Architecture change:**
- **Before (v0.0.6):** 8 input channels per frame (RGB + depth + MV + jitter_x/jitter_y as constant spatial planes). Encoder PixelUnshuffle produced 8×4=32ch, compressed to 16ch. Jitter was learned spatially by the encoder.
- **After (v0.0.8):** 6 input channels per frame (RGB + depth + MV). Jitter passed as separate `jitter_delta [B,2,1,1]` tensor (= `jitter_curr - jitter_prev`), added directly to MV before temporal warp: `mv + jitter_delta`. Encoder PixelUnshuffle now 6×4=24ch → 16ch (~25% fewer params in first layer).
- **Rationale:** Jitter is frame-level metadata, not spatially-varying — broadcasting it to constant planes was wasteful. Compensating MV directly fixes frame alignment at the source (same approach as FSR3/DLSS). During pretrain (MV zeroed), `0 + jitter_delta` provides pure jitter-only warp alignment.

| Parameter | Value |
|-----------|-------|
| Dataset | 1023 bursts (packed .burst) |
| Epochs | 100 (**5 pretrain** + 95 full) |
| Batch size | 4 |
| Patch size | 128 |
| Workers | **40** |
| Model | base_ch=16, temporal_ch=32, groups=4, **6ch input** (was 8ch) |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, perc_out=0, perc_res=0, temporal=0.05, reg=0 |
| Edge boost | 5.0 |
| Device | AmdDevice(0) |
| Log | *(no log available)* |

**Results:**

| Metric | Best |
|--------|------|
| charb (best_loss) | **0.00715** (ep100) |

**Notes:**
- No training log available for this run — only final checkpoint stats recorded.
- best_loss 0.00715 vs v0.0.6's best 0.0062 — slightly higher; expected given the architecture change and reduced pretrain (5 vs 20 epochs).
- First run with jitter-compensated MV architecture. The model now learns temporal fusion with correctly aligned motion vectors rather than relying on the encoder to spatially interpret constant jitter planes.

---

## aa-v0.0.6 (2026-03-21)

**Changes:** Scaled dataset from 410 to 1023 bursts (~2.5×). Same model, hyperparameters, and loss config as v0.0.5. Stopped early at epoch 80 (plateau detected).

| Parameter | Value |
|-----------|-------|
| Dataset | **1023 bursts** (packed .burst) |
| Epochs | **80** (20 pretrain + 60 full), stopped early |
| Batch size | 4 |
| Patch size | 128 |
| Workers | 20 |
| Model | base_ch=16, temporal_ch=32, groups=4 |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, perc_out=0, perc_res=0, temporal=0.05, reg=0 |
| Edge boost | 5.0 |
| Device | AmdDevice(0) |
| Training time | ~781 min (46881s) |
| Log | `aa-train-v0.0.6.log` |

**Results:**

| Metric | Best | Final (ep80) |
|--------|------|--------------|
| psnr_m | 46.3 (ep66) | 45.6 |
| psnr_s | 43.0 (ep66) | 42.0 |
| gap (m-s) | +3.7 (ep75) | +3.6 |
| ssim_m | 0.970 (ep62,66,70,71) | 0.967 |
| charb | 0.0062 (ep62) | 0.0066 |

**Notes:**
- **PSNR improved over v0.0.5** — best 46.3 vs 46.2, final 45.6 vs 45.6 (matched). More data helped peak quality.
- **SSIM did not improve** — plateau at 0.970 vs v0.0.5's 0.976. Larger dataset may require LR decay or more epochs to push SSIM higher.
- **Stopped at epoch 80** — metrics plateaued around ep55–60. Epochs 60–80 showed no improvement trend: PSNR oscillated 45.3–46.3, SSIM 0.965–0.970, charb 0.0062–0.0069.
- **Higher inter-epoch volatility than v0.0.5** — PSNR range ±1.0 dB vs ±0.5 dB. 256 batches/epoch (vs 103) means each epoch sees more diverse data, causing noisier epoch-level metrics.
- **Gap consistently positive** (+3.3 to +3.7) — model output reliably exceeds input quality, comparable to v0.0.5 (+4.0).
- **GPU throttling observed** — epoch time grew from ~510s (ep5–30) to ~650–780s (ep70–80), likely thermal throttling under sustained load.
- Training time ~781 min vs v0.0.5's ~402 min — roughly linear with dataset size × epochs ratio (1023×80 vs 410×100).

---

## aa-v0.0.5 (2026-03-17)

**Changes:** Scaled dataset from 88 to 410 bursts (~4.7×). Same model, hyperparameters, and loss config as v0.0.4.

| Parameter | Value |
|-----------|-------|
| Dataset | **410 bursts** (packed .burst) |
| Epochs | 100 (20 pretrain + 80 full) |
| Batch size | 4 |
| Patch size | 128 |
| Workers | 20 |
| Model | base_ch=16, temporal_ch=32, groups=4 |
| Optimizer | Adam, lr=0.0001 |
| Loss weights | charb_out=1, perc_out=0, perc_res=0, temporal=0.05, reg=0 |
| Edge boost | 5.0 |
| Device | AmdDevice(0) |
| Training time | ~402 min (24110s) |
| Log | `aa-train-v0.0.5.log` |

**Results:**

| Metric | Best | Final (ep100) |
|--------|------|---------------|
| psnr_m | 46.2 (ep70) | 45.6 |
| psnr_s | 42.8 (ep70) | 41.6 |
| gap (m-s) | +4.3 (ep99) | +4.0 |
| ssim_m | 0.979 (ep97) | 0.976 |
| charb | 0.0051 (ep96) | 0.0056 |

**Notes:**
- First run with larger dataset (410 vs 88 bursts). Training time scales roughly linearly (~402 vs ~96 min).
- **Gap (m-s) consistently positive and much larger than prior versions** (+4.0 final vs +2.4 v0.0.4, +1.7 v0.0.3, +2.7 v0.0.2). Model output now clearly and reliably exceeds input quality.
- Best-epoch PSNR (46.2 ep70) and SSIM (0.979 ep97) match or slightly below v0.0.4 peak (47.2/0.977), but final-epoch metrics are more stable: PSNR 45.6 vs 44.5, SSIM 0.976 vs 0.970.
- **Lower inter-epoch volatility** — PSNR range ~44.5–46.2 (±0.9 dB) vs v0.0.4's ~42–47 (±2.5 dB). More data = more stable convergence.
- PRE→FULL transition perfectly smooth: psnr_m 43.5→43.5 (zero dip).
- charb_out final 0.0056 vs v0.0.4's 0.0058 — modest improvement; best 0.0051 vs 0.0048.
- Best PSNR peaked at ep70, not near ep100 — suggests learning rate may need decay for further improvement on this dataset size.

---

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
