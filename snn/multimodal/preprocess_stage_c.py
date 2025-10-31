#!/usr/bin/env python3
"""
Stage C — Multimodal Spike Alignment + Fusion + Normalization

Merges embedding-based video features and mel-spectrogram audio features
from Stage B into unified, normalized multimodal tensors for Stage D.

Output: final_features/<emotion>/<stem>_fusion.npz

usage: (preview)
    python preprocess_stage_c.py --input_root features_preview --output_root final_features_preview --fusion_mode concat --normalize_spikes --threshold 0.1
    
usage: (full)
    python preprocess_stage_c.py

"""

from __future__ import annotations
import argparse, sys, logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

# ---------------- logging ----------------
RESET = "\033[0m"
COLORS = {"INFO": "\033[94m", "WARNING": "\033[93m", "ERROR": "\033[91m"}
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("stage_c")
def clog(level, msg):
    c = COLORS.get(level, "")
    getattr(log, level.lower())(f"{c}{msg}{RESET}")

# ---------------- utils ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

# ---- loaders ----
def load_video_array(npz_path: Path) -> np.ndarray:
    """Load video embeddings from Stage B output files."""
    data = np.load(npz_path, allow_pickle=True)
    if "embeddings" in data:
        return data["embeddings"]
    raise KeyError(f"No key 'embeddings' found in {npz_path}")

def load_audio_array(npz_path: Path) -> np.ndarray:
    """Load mel spectrogram from Stage B audio output."""
    data = np.load(npz_path, allow_pickle=True)
    if "mel" in data:
        return data["mel"]
    raise KeyError(f"No key 'mel' found in {npz_path}")

# ---------------- normalization + threshold ----------------
def normalize_spike_density(x: torch.Tensor, target_rate: float = 0.1) -> torch.Tensor:
    mean_rate = x.mean()
    if mean_rate <= 1e-6:
        return x
    scale = target_rate / mean_rate
    return torch.clamp(x * scale, 0, 1)

def apply_threshold(x: torch.Tensor, thr: float = 0.1) -> torch.Tensor:
    return (x > thr).float()

# ---------------- temporal + fusion ----------------
def temporal_align(video_x: torch.Tensor, audio_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(video_x.shape[1], audio_x.shape[1])

    def pad(x):
        if x.shape[1] == T:
            return x
        pad_len = T - x.shape[1]
        pad_tensor = torch.zeros((*x.shape[:1], pad_len, *x.shape[2:]), device=x.device)
        return torch.cat([x, pad_tensor], dim=1)

    return pad(video_x), pad(audio_x)

def fuse_modalities(video_x: torch.Tensor, audio_x: torch.Tensor, mode: str = "concat") -> torch.Tensor:
    if mode == "concat":
        v_flat = video_x.flatten(start_dim=2)
        a_flat = audio_x.flatten(start_dim=2)
        T = min(v_flat.shape[1], a_flat.shape[1])
        v_flat, a_flat = v_flat[:, :T], a_flat[:, :T]
        return torch.cat([v_flat, a_flat], dim=-1)
    elif mode == "sum":
        T = min(video_x.shape[1], audio_x.shape[1])
        v, a = video_x[:, :T], audio_x[:, :T]
        if v.shape != a.shape:
            a = torch.nn.functional.interpolate(a.unsqueeze(0), size=v.shape[2:], mode="nearest").squeeze(0)
        return v + a
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")

# ---------------- main stage ----------------
def stage_c_merge(
    input_root: Path,
    output_root: Path,
    mode: str = "concat",
    normalize: bool = False,
    threshold: float | None = None,
):
    ensure_dir(output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = []

    emotions = [p for p in input_root.iterdir() if p.is_dir()]
    for emo_dir in emotions:
        emo_name = emo_dir.name
        vid_dir = emo_dir / "video"
        aud_dir = emo_dir / "audio"
        if not vid_dir.exists() or not aud_dir.exists():
            clog("WARNING", f"Skipping {emo_name}: missing modality folder")
            continue

        out_emo_dir = output_root / emo_name
        ensure_dir(out_emo_dir)

        vid_files = {Path(f).stem.replace("_video_feats", ""): f for f in vid_dir.glob("*.npz")}
        aud_files = {Path(f).stem.replace("_audio_feats", ""): f for f in aud_dir.glob("*.npz")}
        aligned_stems = sorted(set(vid_files.keys()) & set(aud_files.keys()))

        if not aligned_stems:
            clog("WARNING", f"No aligned stems found for emotion {emo_name}")
            continue

        for stem in tqdm(aligned_stems, desc=f"{emo_name:10s}", ncols=100):
            try:
                v = to_torch(load_video_array(vid_files[stem]), device)
                a = to_torch(load_audio_array(aud_files[stem]), device)

                if v.ndim == 2: v = v.unsqueeze(0)
                if a.ndim == 2: a = a.unsqueeze(0)

                v_aligned, a_aligned = temporal_align(v, a)

                if normalize:
                    v_aligned = normalize_spike_density(v_aligned)
                    a_aligned = normalize_spike_density(a_aligned)
                if threshold is not None:
                    v_aligned = apply_threshold(v_aligned, threshold)
                    a_aligned = apply_threshold(a_aligned, threshold)

                fused = fuse_modalities(v_aligned, a_aligned, mode)
                fused_np = to_numpy(fused)

                out_path = out_emo_dir / f"{stem}_fusion.npz"
                np.savez_compressed(
                    out_path,
                    video_embeddings=to_numpy(v_aligned),
                    audio_mel=to_numpy(a_aligned),
                    fused=fused_np,
                    emotion=emo_name,
                )
                saved.append(str(out_path))
            except Exception as e:
                clog("ERROR", f"Fusion error for {stem}: {e}")

    clog("INFO", f"✅ Stage C complete — {len(saved)} multimodal fused tensors saved.")
    return saved

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Stage C — Multimodal Spike Fusion")
    parser.add_argument(
        "--input_root",
        type=str,
        default="features",
        help="Root folder containing emotion subfolders with audio/video features",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="final_features",
        help="Destination folder to save fused multimodal features",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="concat",
        choices=["concat", "sum"],
        help="Fusion strategy for modalities",
    )
    parser.add_argument(
        "--normalize_spikes",
        action="store_true",
        help="Normalize spike firing rates before fusion",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold (0–1) to binarize spikes",
    )

    args = parser.parse_args()

    # Convert to Path objects (keeps relative paths unless absolute provided)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    print(f"\n[INFO] 🧩 Stage C configuration")
    print(f" ├─ Input root : {input_root}")
    print(f" ├─ Output root: {output_root}")
    print(f" ├─ Fusion mode: {args.fusion_mode}")
    print(f" ├─ Normalize  : {args.normalize_spikes}")
    print(f" └─ Threshold  : {args.threshold}\n")

    stage_c_merge(
        input_root,
        output_root,
        mode=args.fusion_mode,
        normalize=args.normalize_spikes,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
