#!/usr/bin/env python3
"""
Stage B — GPU feature extraction (non-spiking, ready for Stage C)

Loads caches from cache_dir (both video and audio),
computes:
    - ResNet50 embeddings for video crops
    - Mel-spectrograms for audio
Saves to output_root/<emotion>/<modality>

Usage: (preview)
  python preprocess_stage_b.py --cache_dir data/preview/cache_preview --output_root data/preview/features_preview
  
Usage: (full)
  python preprocess_stage_b.py
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
import torchvision.models as models
import torchaudio
import logging

# ---------------- logging ----------------
RESET = "\033[0m"
COLORS = {"INFO": "\033[94m", "WARNING": "\033[93m", "ERROR": "\033[91m"}
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("stage_b")
def clog(level: str, msg: str):
    c = COLORS.get(level, "")
    getattr(log, level.lower())(f"{c}{msg}{RESET}")

# ---------------- config ----------------
FRAME_SIZE = (224, 224)
AUDIO_SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# ---------------- utilities ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def auto_batch_size_by_vram() -> int:
    if not torch.cuda.is_available():
        return 4
    prop = torch.cuda.get_device_properties(0)
    total_gb = prop.total_memory / (1024 ** 3)
    if total_gb < 5: return 4
    if total_gb < 8: return 8
    if total_gb < 12: return 16
    return 24

# ---------------- model ----------------
class FaceFeatureResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            base = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(2048, 512)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)

# ---------------- audio mel GPU ----------------
def compute_audio_mel_gpu(waveform: np.ndarray, sr: int, device: torch.device):
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    if sr != AUDIO_SR:
        waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SR)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=AUDIO_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    ).to(device)
    mel = mel_spec(waveform)
    db = torchaudio.transforms.AmplitudeToDB().to(device)
    logmel = db(mel)
    return logmel.squeeze(0).cpu().numpy()

# ---------------- stage b processing ----------------
def stage_b_gpu_processing(caches: List[str], out_root: Path, device: torch.device, batch_size: int):
    model = FaceFeatureResNet50().to(device).eval()
    saved = []
    pbar = tqdm(total=len(caches)//2, desc="Stage B (Feature extraction)", ncols=100)

    from collections import defaultdict
    stem_to_files = defaultdict(dict)
    for cf in caches:
        npz = np.load(cf, allow_pickle=True)
        stem = Path(cf).stem
        mod = str(npz.get("modality", "unknown")).lower()
        stem_to_files[stem][mod] = cf
        stem_to_files[stem]["emotion"] = str(npz.get("emotion", "unknown")).lower()

    aligned_stems = [s for s,v in stem_to_files.items() if "video" in v and "audio" in v]

    for stem in aligned_stems:
        files = stem_to_files[stem]
        emo_dir = out_root / files["emotion"]
        vid_dir = emo_dir / "video"
        aud_dir = emo_dir / "audio"
        ensure_dir(vid_dir); ensure_dir(aud_dir)

        # ---------------- Video ----------------
        try:
            npz = np.load(files["video"], allow_pickle=True)
            crops = npz["crops"].astype(np.float32)
            fps = float(npz.get("fps", 24.0))
            feats_list = []

            for i in range(0, len(crops), batch_size):
                batch_imgs = torch.tensor(crops[i:i+batch_size]).permute(0,3,1,2).to(device)
                with torch.no_grad():
                    feats = model(batch_imgs)
                feats_list.append(feats.cpu().numpy())
                del batch_imgs, feats
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            feats_all = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0,512), dtype=np.float32)
            out_path = vid_dir / f"{stem}_video_feats.npz"
            np.savez_compressed(out_path, embeddings=feats_all, fps=fps, emotion=files["emotion"])
            saved.append(str(out_path))
        except Exception as e:
            clog("ERROR", f"[Stage B] Video error for {stem}: {e}")

        # ---------------- Audio ----------------
        try:
            npz = np.load(files["audio"], allow_pickle=True)
            waveform = npz["waveform"]
            sr = int(npz.get("sr", AUDIO_SR))
            mel = compute_audio_mel_gpu(waveform, sr, device)
            out_path = aud_dir / f"{stem}_audio_feats.npz"
            np.savez_compressed(out_path, mel=mel, sr=AUDIO_SR, emotion=files["emotion"])
            saved.append(str(out_path))
        except Exception as e:
            clog("ERROR", f"[Stage B] Audio error for {stem}: {e}")

        pbar.update(1)

    pbar.close()
    clog("INFO", f"✅ Stage B complete — {len(saved)} feature files saved (no spikes).")
    return saved

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage B — Spike Conversion and Feature Extraction"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory containing cached embeddings or features",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="features",
        help="Output directory for processed features",
    )
    parser.add_argument(
        "--batch_override",
        type=int,
        default=0,
        help="Force a specific batch size (0 = auto based on VRAM)",
    )
    args = parser.parse_args()

    # --- device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clog("INFO", f"device={device}")

    # --- prepare output and cache paths ---
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    cache_root = Path(args.cache_dir)
    caches = sorted(str(p) for p in cache_root.rglob("*.npz"))
    if not caches:
        clog("WARNING", f"No cache files found under {cache_root}")
        return

    # --- filter valid modality files ---
    valid = []
    for c in caches:
        try:
            npz = np.load(c, allow_pickle=True)
            mod = str(npz.get("modality", "")).lower()
            emo = npz.get("emotion", None)
            if mod in ("video", "audio") and emo is not None:
                valid.append(c)
        except Exception:
            continue

    caches = valid
    if not caches:
        clog("WARNING", "No valid caches to process")
        return

    # --- batch size determination ---
    batch = args.batch_override if args.batch_override > 0 else auto_batch_size_by_vram()
    clog("INFO", f"Stage B batch size = {batch}")

    # --- main processing ---
    stage_b_gpu_processing(caches, out_root, device, batch)


if __name__ == "__main__":
    main()
