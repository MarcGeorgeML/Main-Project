#!/usr/bin/env python3
"""
Stage A — Multimodal preprocessor (frame & audio normalization only)

Traverses data_sorted/<emotion>/<modality> and caches processed, normalized files.
Each pair of aligned video/audio samples are normalized and saved to cache.

Usage: (preview)
  python preprocess_stage_a.py --input_root data_sorted --cache_dir data/preview/cache_preview --preview 20 --workers 8
  
Usage: (full)
  python preprocess_stage_a.py
"""
from __future__ import annotations
import argparse, os, sys, random, logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing as mp
import soundfile as sf
import collections
import tempfile

# ---------------- Logging ----------------
RESET = "\033[0m"
COLORS = {"INFO": "\033[94m", "WARNING": "\033[93m", "ERROR": "\033[91m", "DEBUG": "\033[90m"}
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("stage_a")
def clog(level: str, msg: str):
    c = COLORS.get(level, "")
    getattr(log, level.lower())(f"{c}{msg}{RESET}")

# ---------------- Config ----------------
TARGET_FPS = 24
FRAME_SIZE = (224, 224)
MAX_TASKS_PER_CHILD = 20

# ---------------- Utils ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_video_frames_opencv(video_path: Path, target_fps: int = TARGET_FPS) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], target_fps
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(src_fps / target_fps))) if src_fps > 0 else 1
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, src_fps

def resample_frames(frames: List[np.ndarray], src_fps: float, target_fps: int = TARGET_FPS):
    if not frames or abs(src_fps - target_fps) < 1e-3:
        return frames, src_fps
    duration = len(frames) / src_fps
    desired_n = max(1, int(round(duration * target_fps)))
    idxs = np.linspace(0, len(frames) - 1, desired_n).astype(int)
    return [frames[i] for i in idxs], target_fps

# ---------------- Normalization utilities ----------------
def normalize_image(img: np.ndarray):
    """Normalize image to [0,1] range — not spike domain yet."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)

def normalize_audio_rms(audio_data: np.ndarray):
    """Normalize audio waveform to unit RMS (for consistent feature extraction)."""
    rms = np.sqrt(np.mean(np.square(audio_data))) + 1e-8
    return (audio_data / rms).astype(np.float32)

# ---------------- Worker ----------------
def stage_a_worker(task: Tuple[Path, str, str, Path]) -> Tuple[Path, dict]:
    file_path, modality, emotion, cache_dir = task
    cache_dir = Path(cache_dir) / modality
    ensure_dir(cache_dir)
    cache_file = cache_dir / f"{file_path.stem}.npz"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz", dir=cache_dir) as tmp:
            tmp_path = Path(tmp.name)

        if modality == "video":
            frames, src_fps = read_video_frames_opencv(file_path)
            if not frames:
                return None, {"error": "no_frames"}
            frames, used_fps = resample_frames(frames, src_fps, TARGET_FPS)

            crops = [normalize_image(cv2.resize(frm, FRAME_SIZE)) for frm in frames]
            np.savez_compressed(
                tmp_path,
                modality="video",
                video_path=str(file_path),
                emotion=emotion,
                fps=float(used_fps),
                n_frames=len(crops),
                crops=np.array(crops, dtype=np.float32),
            )

        elif modality == "audio":
            data, sr = sf.read(str(file_path))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            norm_audio = normalize_audio_rms(data)
            np.savez_compressed(
                tmp_path,
                modality="audio",
                audio_path=str(file_path),
                emotion=emotion,
                sr=sr,
                waveform=norm_audio,
                duration=len(norm_audio)/sr,
            )
        else:
            return None, {"error": f"unknown modality {modality}"}

        os.replace(tmp_path, cache_file)
        return cache_file, {}

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return None, {"error": str(e)}

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage A — Preprocess raw audio/video into normalized cache files"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="data_sorted",
        help="Root directory containing sorted data folders (emotion/video/audio)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to store normalized cache files before feature extraction",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Number of samples to process for quick preview (0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) - 4),
        help="Number of parallel worker processes (defaults to CPU count minus 4)",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    cache_root = Path(args.cache_dir)
    ensure_dir(cache_root)

    entries: list[tuple[Path, str, str, Path]] = []

    # --- build mapping: stem -> {video, audio, emotion} ---
    stem_to_files = collections.defaultdict(dict)
    for emo_dir in [p for p in input_root.iterdir() if p.is_dir()]:
        emo_name = emo_dir.name
        for modality in ["video", "audio"]:
            mod_dir = emo_dir / modality
            if not mod_dir.exists():
                continue
            for f in mod_dir.iterdir():
                if f.is_file():
                    stem_to_files[f.stem][modality] = f
                    stem_to_files[f.stem]["emotion"] = emo_name

    # --- keep only aligned pairs ---
    valid_stems = [s for s, v in stem_to_files.items() if "video" in v and "audio" in v]
    if args.preview > 0:
        valid_stems = random.sample(valid_stems, min(args.preview, len(valid_stems)))

    for stem in valid_stems:
        emo = stem_to_files[stem]["emotion"]
        entries.append((stem_to_files[stem]["video"], "video", emo, cache_root))
        entries.append((stem_to_files[stem]["audio"], "audio", emo, cache_root))

    if not entries:
        clog("WARNING", "No aligned audio-video pairs found to process.")
        return

    ensure_dir(cache_root / "video")
    ensure_dir(cache_root / "audio")

    # --- multiprocessing ---
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        results = list(tqdm(pool.imap_unordered(stage_a_worker, entries), total=len(entries)))

    valid = [r[0] for r in results if r and r[0]]
    clog("INFO", f"✅ Stage A complete — {len(valid)} normalized caches created (pre-feature, pre-spike).")


if __name__ == "__main__":
    mp.freeze_support()
    main()
