# Text-to-Image Diffusion Model — From Scratch

A from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch,
extended to support text-conditioned image generation via CLIP embeddings + cross-attention.

No pretrained diffusion models or high-level `diffusers` shortcuts are used for the core
forward/reverse process, U-Net, or sampling logic. The goal is to demonstrate understanding
of the underlying math and engineering, not just API usage.

## Status

**Phase 1 complete — ready for GPU training on Kaggle.**

## Project Phases

### Phase 1 — Unconditional DDPM (MNIST / Fashion-MNIST)
- [x] Forward diffusion process (`src/utils/diffusion.py`) — linear beta schedule, closed-form `q_sample`
- [x] U-Net denoiser (`src/models/unet.py`) — sinusoidal time embedding, ResBlock + GroupNorm + SiLU, skip connections, 11.6M params
- [x] Dataset loader (`src/data/dataset.py`) — Fashion-MNIST / MNIST via torchvision, normalised to [-1, 1]
- [x] Training loop (`scripts/train.py`) — epsilon-prediction loss, CSV loss log, checkpointing, sample grids every N epochs
- [x] DDPM ancestral sampling (`src/utils/sampling.py`) — precomputed coefficients, both variance choices documented
- [x] Sampling CLI (`scripts/sample.py`) — load checkpoint, run reverse process, save PNG grid

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
│   ├── data/          # Dataset loaders (MNIST, Fashion-MNIST, Flowers102)
│   └── utils/         # Diffusion scheduler, sampling, EMA, FID, visualization
├── configs/           # YAML configs for each phase/experiment
├── scripts/           # train.py, sample.py, evaluate.py
├── notebooks/         # Kaggle/Colab notebooks for GPU training
├── outputs/           # checkpoints/, samples/, logs/  (gitignored)
└── docs/              # math derivations, write-ups
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Full training run (set device: cuda in config for Kaggle GPU)
python scripts/train.py --config configs/phase1_unconditional.yaml

# Override device for local CPU debugging (tiny batch)
python scripts/train.py --config configs/phase1_unconditional.yaml --device cpu
```

Checkpoints are saved to `outputs/checkpoints/`, sample grids to `outputs/samples/`, and a live loss CSV + plot to `outputs/logs/` — all every `save_every` / `sample_every` epochs (configured in the YAML).

## Sampling

```bash
# Generate images from a checkpoint
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt

# Override number of samples, grid layout, and output path
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt \
    --n-samples 64 --nrow 8 --output my_grid.png

# Self-test without a real checkpoint
python scripts/sample.py --sanity-check
```

## Compute Plan

- Development/debugging: local (CPU, tiny batches)
- Actual training: Kaggle Notebooks (free GPU, 30hr/week quota)
