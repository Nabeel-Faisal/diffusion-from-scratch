# Text-to-Image Diffusion Model — From Scratch

A from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch,
extended to support text-conditioned image generation via CLIP embeddings + cross-attention.

No pretrained diffusion models or high-level `diffusers` shortcuts are used for the core
forward/reverse process, U-Net, or sampling logic. The goal is to demonstrate understanding
of the underlying math and engineering, not just API usage.

## Project Phases

### Phase 1 — Unconditional DDPM (MNIST / Fashion-MNIST)
- [ ] Forward diffusion process (noise scheduler, linear beta schedule)
- [ ] U-Net denoiser (time embedding via sinusoidal positional encoding)
- [ ] Training loop with epsilon-prediction (noise prediction) loss
- [ ] DDPM ancestral sampling
- [ ] Training curves (loss vs. step) + sample grids over epochs

### Phase 2 — Improved Sampling & Evaluation
- [ ] Cosine noise schedule
- [ ] DDIM sampling (deterministic, faster)
- [ ] FID score computation
- [ ] Ablation: linear vs cosine schedule, DDPM vs DDIM steps

### Phase 3 — Text Conditioning (Flowers102, 64x64)
- [ ] CLIP text encoder integration (frozen, pretrained — only this part uses an external model)
- [ ] Cross-attention conditioning in U-Net
- [ ] Classifier-free guidance (CFG)
- [ ] Train on captioned dataset, evaluate guidance scale effect

### Phase 4 — Deliverables
- [ ] Math write-up (forward/reverse process, ELBO derivation, loss simplification)
- [ ] Gradio demo (text prompt -> generated image)
- [ ] Comparison with pretrained Stable Diffusion outputs
- [ ] Polished README with results, training curves, sample images

## Repo Structure

```
diffusion-from-scratch/
├── src/
│   ├── models/        # U-Net, attention blocks, time embeddings, CLIP wrapper
│   ├── data/           # Dataset loaders (MNIST, Fashion-MNIST, Flowers102)
│   └── utils/          # Diffusion scheduler, EMA, FID, visualization helpers
├── configs/             # YAML configs for each phase/experiment
├── scripts/             # train.py, sample.py, evaluate.py
├── notebooks/           # Kaggle/Colab notebooks for GPU training
├── outputs/
│   ├── checkpoints/
│   ├── samples/         # generated image grids per epoch
│   └── logs/            # training logs, loss curves
└── docs/                # math derivations, write-ups
```

## Compute Plan

- Development/debugging: local (CPU, tiny batches) via Claude Code
- Actual training: Kaggle Notebooks (free GPU, 30hr/week quota)

## Setup

```bash
pip install -r requirements.txt
```

## Status

🚧 In progress — Phase 1
