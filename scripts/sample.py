"""
Generate images from a trained DDPM checkpoint.

Supports both Phase 1/2 unconditional checkpoints and Phase 3 text-conditioned
checkpoints.  The model type is auto-detected from the embedded config:
  - config.model.text_emb_dim is None  → unconditional (Phase 1/2)
  - config.model.text_emb_dim == 512   → text-conditioned (Phase 3)

For conditional checkpoints, pass --prompts to run CFG-DDIM generation.
A caption .txt file is written alongside the PNG listing which image index
corresponds to which prompt.

EMA weights are preferred over raw model weights when both are present in
the checkpoint (Phase 3 checkpoints always include EMA).

Usage:
    # Phase 1/2 — unconditional
    python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt

    # Phase 3 — text-conditioned with CFG
    python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0050.pt \\
        --prompts "a photo of a dog" "a photo of an airplane" "a photo of a cat"

    # Phase 3 — override guidance scale and output path
    python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0050.pt \\
        --prompts "a photo of a ship" --guidance-scale 3.0 --output ship.png

    # Phase 3 — generate all 10 CIFAR-10 classes
    python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0050.pt \\
        --prompts \\
          "a photo of a airplane" "a photo of a automobile" "a photo of a bird" \\
          "a photo of a cat" "a photo of a deer" "a photo of a dog" \\
          "a photo of a frog" "a photo of a horse" "a photo of a ship" \\
          "a photo of a truck" \\
        --nrow 5

    # Self-test (no checkpoint needed)
    python scripts/sample.py --sanity-check
"""

import argparse
import copy
import math
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unet import UNet
from src.utils.diffusion import NoiseScheduler
from src.utils.sampling import SamplingCoeffs, ddim_cfg_sample, ddim_sample, ddpm_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> torch.device:
    """Return the best available device, falling back gracefully."""
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[WARN] CUDA not available — falling back to MPS.")
            return torch.device("mps")
        print("[WARN] CUDA not available — falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[WARN] MPS not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _save_grid(samples: torch.Tensor, path: Path, nrow: int) -> None:
    """Rescale samples from [-1, 1] to [0, 1] and write a PNG grid."""
    grid = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(grid, path, nrow=nrow)


def _save_captions(
    prompts: List[str],
    path: Path,
    checkpoint_name: str,
    guidance_scale: float,
) -> None:
    """
    Write a plain-text caption file co-located with the output PNG.

    Format:
        # checkpoint: epoch_0050.pt
        # guidance_scale: 7.5
        # images: 3

        image_00: a photo of a dog
        image_01: a photo of an airplane
        image_02: a photo of a cat
    """
    with open(path, "w") as f:
        f.write(f"# checkpoint: {checkpoint_name}\n")
        f.write(f"# guidance_scale: {guidance_scale}\n")
        f.write(f"# images: {len(prompts)}\n\n")
        for i, prompt in enumerate(prompts):
            f.write(f"image_{i:02d}: {prompt}\n")


def _is_conditional(cfg: dict) -> bool:
    """Return True if the config describes a text-conditioned (Phase 3) model."""
    return cfg["model"].get("text_emb_dim") is not None


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate(
    checkpoint_path: Path,
    prompts: Optional[List[str]] = None,
    guidance_scale: Optional[float] = None,
    n_samples: Optional[int] = None,
    device_str: Optional[str] = None,
    output_path: Optional[Path] = None,
    nrow: Optional[int] = None,
    _clip_enc=None,
) -> Path:
    """
    Load a checkpoint and generate images, dispatching between unconditional
    and CFG-DDIM generation based on the model type and supplied prompts.

    Dispatch rules:
      - Unconditional checkpoint + no prompts  → DDPM/DDIM (Phase 1/2 behaviour)
      - Conditional checkpoint  + prompts      → CFG-DDIM  (Phase 3)
      - Conditional checkpoint  + no prompts   → unconditional DDIM (text_emb=None)
      - Unconditional checkpoint + prompts     → ValueError (not supported)

    EMA weights are preferred when the checkpoint contains both
    ``model_state_dict`` and a non-None ``ema_state_dict``.

    Args:
        checkpoint_path: Path to a .pt file saved by scripts/train.py.
        prompts:         List of caption strings for conditional generation.
                         One image is generated per prompt.
        guidance_scale:  CFG weight w in ε_cfg = (1+w)·ε_cond − w·ε_uncond.
                         Overrides the checkpoint config; defaults to 7.5.
        n_samples:       Images to generate (unconditional path only).
                         Defaults to config training.num_samples.
        device_str:      Override device ('cpu', 'cuda', 'mps').
        output_path:     Where to write the PNG.  Auto-derived if None.
        nrow:            Images per row.  Defaults to floor(sqrt(n_images)).
        _clip_enc:       Inject a mock CLIP encoder (for testing only).

    Returns:
        Path of the saved PNG.

    Raises:
        ValueError: if prompts are supplied with an unconditional checkpoint.
    """
    # --- Load checkpoint ---
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    epoch = ckpt.get("epoch", 0)

    conditional = _is_conditional(cfg)

    # Validate: prompts require a conditional model.
    if prompts and not conditional:
        raise ValueError(
            "--prompts requires a text-conditioned checkpoint "
            "(config.model.text_emb_dim must be set). "
            "This checkpoint is unconditional (Phase 1/2). "
            "Run without --prompts, or supply a Phase 3 checkpoint."
        )

    # --- Resolve runtime parameters ---
    device = _resolve_device(device_str or cfg["training"]["device"])

    # n_samples: number of images to generate.
    # For prompted generation, len(prompts) always wins.
    if prompts:
        n = len(prompts)
    elif n_samples is not None:
        n = n_samples
    else:
        n = cfg["training"].get("num_samples", 16)

    # guidance_scale: CLI > checkpoint config > 7.5 default.
    gs = (
        guidance_scale
        if guidance_scale is not None
        else cfg.get("sampling", {}).get("guidance_scale", 7.5)
    )

    grid_nrow = nrow if nrow is not None else max(1, int(math.floor(n ** 0.5)))

    # Auto-derive output path: suffix "_prompted" when prompts are supplied
    # so prompted and unconditional runs don't clobber each other.
    if output_path is None:
        paths = cfg.get("paths", {})
        sample_dir = Path(paths.get("sample_dir", "outputs/samples"))
        sample_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_prompted" if prompts else ""
        output_path = sample_dir / f"sampled_epoch_{epoch:04d}{suffix}.png"

    # --- Build noise scheduler ---
    scheduler = NoiseScheduler(
        timesteps  = cfg["diffusion"]["timesteps"],
        beta_start = cfg["diffusion"].get("beta_start", 1e-4),
        beta_end   = cfg["diffusion"].get("beta_end", 0.02),
        schedule   = cfg["diffusion"].get("beta_schedule", "linear"),
        cosine_s   = cfg["diffusion"].get("cosine_s", 0.008),
    ).to(device)

    # --- Build U-Net ---
    model = UNet(
        img_channels  = cfg["dataset"]["channels"],
        base_channels = cfg["model"]["base_channels"],
        channel_mults = tuple(cfg["model"]["channel_mults"]),
        num_res_blocks= cfg["model"]["num_res_blocks"],
        time_emb_dim  = cfg["model"]["time_emb_dim"],
        text_emb_dim  = cfg["model"].get("text_emb_dim"),
        attn_heads    = cfg["model"].get("attn_heads", 8),
    ).to(device)

    # Load weights — prefer EMA when available (Phase 3 checkpoints always have it).
    if ckpt.get("ema_state_dict") is not None:
        model.load_state_dict(ckpt["ema_state_dict"])
        weights_used = "EMA"
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        weights_used = "model"

    # --- Load CLIP (Phase 3 + prompts only) ---
    clip_enc_to_use = None
    if conditional and prompts:
        if _clip_enc is not None:
            clip_enc_to_use = _clip_enc
        else:
            from src.models.clip_encoder import CLIPTextEncoder
            clip_model_name = cfg.get("clip", {}).get("model_name", "ViT-B/32")
            print(f"Loading CLIP ({clip_model_name}) …")
            clip_enc_to_use = CLIPTextEncoder(clip_model_name, device).to(device)

    # --- Sampling parameters ---
    img_shape    = (cfg["dataset"]["channels"],
                    cfg["dataset"]["image_size"],
                    cfg["dataset"]["image_size"])
    sampler_type = cfg["training"].get("sampler", "ddpm")
    ddim_steps   = cfg["training"].get("ddim_steps", 50)
    ddim_eta     = cfg["training"].get("ddim_eta", 0.0)

    # --- Print run summary ---
    mode = "conditional (CFG-DDIM)" if (prompts and conditional) else "unconditional"
    print(f"Checkpoint : {checkpoint_path.name}  (epoch {epoch}, weights={weights_used})")
    print(f"Device     : {device}")
    print(f"Mode       : {mode}")
    print(f"Schedule   : {cfg['diffusion'].get('beta_schedule', 'linear')}")
    if prompts and conditional:
        print(f"Sampler    : ddim ({ddim_steps} steps, η={ddim_eta}, "
              f"guidance_scale={gs})")
    elif sampler_type == "ddim":
        print(f"Sampler    : ddim ({ddim_steps} steps, η={ddim_eta})")
    else:
        print(f"Sampler    : ddpm")
    print(f"Generating : {n} image{'s' if n != 1 else ''}  |  "
          f"grid {grid_nrow} per row  |  shape {img_shape}")
    if prompts:
        print("Prompts:")
        for i, p in enumerate(prompts):
            print(f"  [{i:02d}] {p}")

    # --- Generate ---
    if prompts and conditional and clip_enc_to_use is not None:
        samples = ddim_cfg_sample(
            model          = model,
            clip_enc       = clip_enc_to_use,
            prompts        = prompts,
            scheduler      = scheduler,
            device         = device,
            img_shape      = img_shape,
            n_steps        = ddim_steps,
            eta            = ddim_eta,
            guidance_scale = gs,
        )
    elif sampler_type == "ddim" or conditional:
        # Conditional model with no prompts: unconditional DDIM (text_emb=None).
        # DDIM is also the default for Phase 3 checkpoints.
        samples = ddim_sample(
            model, scheduler, n_samples=n, img_shape=img_shape,
            device=device, n_steps=ddim_steps, eta=ddim_eta, show_progress=True,
        )
    else:
        samples = ddpm_sample(
            model, scheduler, n_samples=n, img_shape=img_shape,
            device=device, show_progress=True, coeffs=SamplingCoeffs(scheduler),
        )

    # --- Save outputs ---
    _save_grid(samples, output_path, nrow=grid_nrow)
    print(f"Saved      : {output_path}")

    if prompts:
        caption_path = output_path.with_suffix(".txt")
        _save_captions(prompts, caption_path, checkpoint_path.name, gs)
        print(f"Captions   : {caption_path}")

    return output_path


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check() -> None:
    """
    End-to-end tests using tiny fake checkpoints — runs in seconds on CPU.
    CLIP is never loaded; a mock encoder is injected via _clip_enc.

    Checks:
      1. Unconditional checkpoint + no prompts → generates PNG (Phase 1/2 compat).
      2. Auto-derived filename uses epoch from the checkpoint.
      3. Default nrow = floor(sqrt(n_samples)).
      4. Conditional checkpoint + prompts + mock CLIP → CFG-DDIM dispatch.
      5. Caption .txt file written co-located with the PNG.
      6. Caption file contains all prompts with correct image indices.
      7. Conditional checkpoint + no prompts → unconditional DDIM (no CLIP loaded).
      8. Prompts + unconditional checkpoint → ValueError.
      9. EMA state dict preferred over model state dict when both are present.
    """
    T = 20

    # --- Tiny unconditional config (Phase 1/2) ---
    cfg_uncond = {
        "diffusion": {
            "timesteps": T, "beta_start": 1e-4,
            "beta_end": 0.02, "beta_schedule": "linear",
        },
        "model": {
            "base_channels": 32, "channel_mults": [1, 2],
            "num_res_blocks": 1, "time_emb_dim": 64,
        },
        "dataset": {
            "name": "fashion_mnist", "image_size": 28,
            "channels": 1, "batch_size": 4,
        },
        "training": {
            "epochs": 1, "learning_rate": 2e-4, "device": "cpu",
            "save_every": 1, "sample_every": 1, "num_samples": 4,
        },
        "paths": {
            "checkpoint_dir": "outputs/checkpoints",
            "sample_dir":     "outputs/samples",
            "log_dir":        "outputs/logs",
        },
    }

    # --- Tiny conditional config (Phase 3) ---
    cfg_cond = {
        "diffusion": {
            "timesteps": T, "beta_start": 1e-4,
            "beta_end": 0.02, "beta_schedule": "linear",
        },
        "model": {
            "base_channels": 32, "channel_mults": [1, 2],
            "num_res_blocks": 1, "time_emb_dim": 64,
            "text_emb_dim": 512, "attn_heads": 8,
        },
        "dataset": {
            "name": "cifar10", "image_size": 32,
            "channels": 3, "batch_size": 4,
        },
        "training": {
            "epochs": 1, "learning_rate": 2e-4, "device": "cpu",
            "save_every": 1, "sample_every": 1, "num_samples": 4,
            "sampler": "ddim", "ddim_steps": 3, "ddim_eta": 0.0,
        },
        "sampling": {"guidance_scale": 7.5},
        "paths": {
            "checkpoint_dir": "outputs/checkpoints",
            "sample_dir":     "outputs/samples",
            "log_dir":        "outputs/logs",
        },
    }

    # Tiny models
    model_uncond = UNet(
        img_channels=1, base_channels=32, channel_mults=(1, 2),
        num_res_blocks=1, time_emb_dim=64,
    )
    model_cond = UNet(
        img_channels=3, base_channels=32, channel_mults=(1, 2),
        num_res_blocks=1, time_emb_dim=64, text_emb_dim=512, attn_heads=8,
    )

    # Mock CLIP encoder: returns random unit-normed vectors, no weights downloaded.
    class _MockCLIP:
        def __call__(self, captions: List[str]) -> torch.Tensor:
            vecs = torch.randn(len(captions), 512)
            return vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    mock_clip = _MockCLIP()

    # Save fake checkpoints to temp files.
    ckpt_uncond_path = Path(tempfile.mktemp(suffix=".pt"))
    ckpt_cond_path   = Path(tempfile.mktemp(suffix=".pt"))
    torch.save(
        {"epoch": 7, "model_state_dict": model_uncond.state_dict(),
         "ema_state_dict": None, "optimizer_state_dict": {}, "config": cfg_uncond},
        ckpt_uncond_path,
    )
    torch.save(
        {"epoch": 50, "model_state_dict": model_cond.state_dict(),
         "ema_state_dict": None, "optimizer_state_dict": {}, "config": cfg_cond},
        ckpt_cond_path,
    )

    out_dir = Path(tempfile.mkdtemp())

    print("=== Unconditional path (Phase 1/2 backward compat) ===")

    # 1. Unconditional checkpoint, no prompts → generates PNG
    out1 = out_dir / "uncond.png"
    result = generate(ckpt_uncond_path, n_samples=4, device_str="cpu",
                      output_path=out1, nrow=2)
    assert result == out1 and out1.exists() and out1.stat().st_size > 0
    print(f"[PASS] 1. Unconditional generate → {out1.stat().st_size} bytes")

    # 2. Auto-derived filename uses epoch
    auto = generate(ckpt_uncond_path, n_samples=4, device_str="cpu", nrow=2)
    assert auto.name == "sampled_epoch_0007.png", f"Got: {auto.name}"
    print(f"[PASS] 2. Auto filename: {auto.name}")
    auto.unlink(missing_ok=True)

    # 3. Default nrow = floor(sqrt(n))
    for n, expected in [(1, 1), (4, 2), (7, 2), (9, 3), (16, 4)]:
        got = max(1, int(math.floor(n ** 0.5)))
        assert got == expected, f"nrow({n}) = {got}, expected {expected}"
    print("[PASS] 3. Default nrow = floor(sqrt(n_samples))")

    print("\n=== Conditional dispatch (Phase 3 with mock CLIP) ===")

    prompts = ["a photo of a cat", "a photo of a dog", "a photo of a ship"]

    # 4. Conditional + prompts → CFG dispatch, generates PNG
    out4 = out_dir / "cond_prompted.png"
    result4 = generate(ckpt_cond_path, prompts=prompts, guidance_scale=7.5,
                       device_str="cpu", output_path=out4, _clip_enc=mock_clip)
    assert result4 == out4 and out4.exists() and out4.stat().st_size > 0
    print(f"[PASS] 4. Conditional + prompts → CFG-DDIM PNG ({out4.stat().st_size} bytes)")

    # 5. Caption .txt written alongside PNG
    cap_path = out4.with_suffix(".txt")
    assert cap_path.exists(), "Caption .txt file not created"
    print(f"[PASS] 5. Caption file exists: {cap_path.name}")

    # 6. Caption file format: contains all prompts with correct indices
    caption_text = cap_path.read_text()
    for i, p in enumerate(prompts):
        expected_line = f"image_{i:02d}: {p}"
        assert expected_line in caption_text, \
            f"Missing line '{expected_line}' in caption file"
    assert "# guidance_scale: 7.5" in caption_text
    print(f"[PASS] 6. Caption file contains all {len(prompts)} prompts with indices")

    # 7. Conditional + no prompts → unconditional DDIM (no CLIP loaded)
    out7 = out_dir / "cond_uncond.png"
    result7 = generate(ckpt_cond_path, n_samples=2, device_str="cpu",
                       output_path=out7)
    assert result7 == out7 and out7.exists() and out7.stat().st_size > 0
    assert not out7.with_suffix(".txt").exists(), \
        "Caption file should not be created without prompts"
    print(f"[PASS] 7. Conditional + no prompts → unconditional DDIM (no CLIP, no .txt)")

    print("\n=== Edge cases ===")

    # 8. Prompts + unconditional checkpoint → ValueError
    raised = False
    try:
        generate(ckpt_uncond_path, prompts=["a photo of a cat"],
                 device_str="cpu", output_path=out_dir / "err.png",
                 _clip_enc=mock_clip)
    except ValueError as e:
        raised = True
        assert "unconditional" in str(e).lower() or "text_emb_dim" in str(e).lower()
    assert raised, "Expected ValueError for prompts + unconditional checkpoint"
    print("[PASS] 8. Prompts + unconditional checkpoint → ValueError")

    # 9. EMA state dict preferred when present and non-None
    model_a = UNet(img_channels=1, base_channels=32, channel_mults=(1, 2),
                   num_res_blocks=1, time_emb_dim=64)
    model_b = copy.deepcopy(model_a)
    with torch.no_grad():
        for p in model_b.parameters():
            p.fill_(0.12345)

    ckpt_ema_path = Path(tempfile.mktemp(suffix=".pt"))
    torch.save(
        {"epoch": 99, "model_state_dict": model_a.state_dict(),
         "ema_state_dict": model_b.state_dict(),
         "optimizer_state_dict": {}, "config": cfg_uncond},
        ckpt_ema_path,
    )
    loaded = torch.load(ckpt_ema_path, map_location="cpu", weights_only=False)
    check = UNet(img_channels=1, base_channels=32, channel_mults=(1, 2),
                 num_res_blocks=1, time_emb_dim=64)
    if loaded.get("ema_state_dict") is not None:
        check.load_state_dict(loaded["ema_state_dict"])
        used = "EMA"
    else:
        check.load_state_dict(loaded["model_state_dict"])
        used = "model"
    first_p = next(check.parameters())
    assert torch.allclose(first_p, torch.full_like(first_p, 0.12345), atol=1e-4), \
        "EMA weights not loaded (expected all-0.12345)"
    assert used == "EMA"
    print("[PASS] 9. EMA state dict preferred over model state dict when both present")

    # Cleanup
    for path in [ckpt_uncond_path, ckpt_cond_path, ckpt_ema_path]:
        path.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)

    print("\nAll sanity checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample images from a trained DDPM checkpoint "
                    "(Phase 1/2 unconditional or Phase 3 text-conditioned)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint .pt file (saved by scripts/train.py)",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=None, metavar="PROMPT",
        help=(
            "One or more text prompts for conditional generation (Phase 3 only). "
            "Generates one image per prompt. "
            "Example: --prompts \"a photo of a dog\" \"a photo of a ship\""
        ),
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=None, dest="guidance_scale",
        help=(
            "CFG guidance scale w (Phase 3 only). "
            "ε_cfg = (1+w)·ε_cond − w·ε_uncond. "
            "Default: value from checkpoint config, or 7.5."
        ),
    )
    parser.add_argument(
        "--n-samples", type=int, default=None, dest="n_samples",
        help="Number of images to generate — unconditional path only "
             "(prompted path always generates len(prompts) images)",
    )
    parser.add_argument(
        "--nrow", type=int, default=None,
        help="Images per row in the output grid (default: floor(sqrt(n_images))). "
             "Tip: use --nrow 5 for a 2×5 grid of 10 CIFAR-10 classes.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path. Default: outputs/samples/sampled_epoch_NNNN[_prompted].png",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device: cpu / cuda / mps",
    )
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Run self-test without a real checkpoint and exit",
    )
    args = parser.parse_args()

    if args.sanity_check:
        _sanity_check()
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required (or use --sanity-check)")
        generate(
            checkpoint_path = Path(args.checkpoint),
            prompts         = args.prompts,
            guidance_scale  = args.guidance_scale,
            n_samples       = args.n_samples,
            device_str      = args.device,
            output_path     = Path(args.output) if args.output else None,
            nrow            = args.nrow,
        )
