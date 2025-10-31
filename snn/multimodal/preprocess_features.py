#!/usr/bin/env python3
"""
preprocess_features_fastpreview.py — Uses Mediapipe Tasks FaceLandmarker (new API)
Optimized for extracting facial landmarks, crops, and audio mels efficiently.
No re-encoding required.

Usage:
  for previewing a small subset of data:
  python preprocess_features.py --input_root data_sorted --output_root features_preview --cache_dir cache_preview --run_stage both --preview 20
  
  for full dataset processing:
  python preprocess_features.py --input_root data_sorted --output_root features --cache_dir cache --run_stage both
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import av
import soundfile as sf
import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio
from tqdm import tqdm
import multiprocessing as mp
import random
import pandas as pd
from PIL import Image  # ✅ For general image handling if needed

# ✅ Mediapipe Tasks API — renamed to avoid "mp" conflict
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- Config ----------------
TARGET_FPS = 24
FRAME_SIZE = (224, 224)
AUDIO_SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
CACHE_SUFFIX = "_cache.npz"
DONE_SUFFIX = ".done"
MAXTASKSPERCHILD = 5

# ---------------- Utils ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_cpu_worker_count():
    return max(1, (os.cpu_count() or 8) - 4)

def auto_batch_size_by_vram():
    if not torch.cuda.is_available():
        return 4
    prop = torch.cuda.get_device_properties(0)
    total_gb = prop.total_memory / (1024 ** 3)
    if total_gb < 5: return 4
    if total_gb < 8: return 8
    if total_gb < 12: return 16
    return 24

# ---------------- Video helpers ----------------
def read_video_frames(video_path: Path, target_fps: int = TARGET_FPS):
    try:
        container = av.open(str(video_path))
        vstream = next((s for s in container.streams if s.type == "video"), None)
        src_fps = float(vstream.average_rate) if vstream and vstream.average_rate else target_fps
        frames = []
        step = max(1, int(round(src_fps / target_fps))) if src_fps > 0 else 1
        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0:
                frames.append(frame.to_ndarray(format="bgr24"))
        container.close()
        return frames, src_fps
    except Exception as e:
        print(f"[⚠ PyAV] {video_path.name} failed: {e} — fallback OpenCV")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[❌ OpenCV] cannot open {video_path}")
            return [], float(target_fps)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        frames, idx, step = [], 0, max(1, int(round(src_fps / target_fps)))
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frm)
            idx += 1
        cap.release()
        return frames, src_fps

def resample_frames(frames, src_fps, target_fps=TARGET_FPS):
    if not frames or abs(src_fps - target_fps) < 1e-3:
        return frames, src_fps
    duration = len(frames) / src_fps
    desired_n = max(1, int(round(duration * target_fps)))
    idxs = np.linspace(0, len(frames) - 1, desired_n).astype(int)
    return [frames[i] for i in idxs], float(target_fps)

# ---------------- FaceLandmarker (New Mediapipe Tasks API) ----------------
_landmarker = None

def get_facelandmarker():
    global _landmarker
    if _landmarker is None:
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"⚠️ Missing model file: {model_path}\n"
                f"Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker

def extract_landmarks_mediapipe(frame):
    landmarker = get_facelandmarker()
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w, _ = frame.shape
    lm = result.face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return pts

def make_heatmap_from_landmarks(landmarks, size=FRAME_SIZE, sigma=3):
    W, H = size
    hm = np.zeros((H, W), np.float32)
    if landmarks is None or len(landmarks) == 0:
        return hm
    for (x, y) in landmarks:
        cx = int(np.clip(x * W, 0, W - 1))
        cy = int(np.clip(y * H, 0, H - 1))
        cv2.circle(hm, (cx, cy), 1, 1.0, -1)
    hm = cv2.GaussianBlur(hm, (7, 7), sigma)
    return hm / (hm.max() + 1e-8)

# ---------------- Stage A (CPU) ----------------
def stage_a_worker(task):
    file_path, modality, emotion, cache_dir = task
    file_path = Path(file_path)
    cache_file = Path(cache_dir) / f"{file_path.stem}{CACHE_SUFFIX}"
    done_file = Path(cache_dir) / f"{file_path.stem}{DONE_SUFFIX}"

    if done_file.exists() and cache_file.exists():
        return str(cache_file), {'skipped': True}

    try:
        if modality == "video":
            frames, src_fps = read_video_frames(file_path)
            if not frames:
                return None, {'error': 'no_frames'}
            frames, used_fps = resample_frames(frames, src_fps, TARGET_FPS)

            crops, heatmaps, lmarks = [], [], []
            for frm in frames:
                lm = extract_landmarks_mediapipe(frm)
                if lm is None:
                    crop = cv2.resize(frm, FRAME_SIZE)
                    crops.append(crop)
                    heatmaps.append(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), np.float32))
                    lmarks.append(np.zeros((478, 2), np.float32))
                    continue

                minx, maxx = lm[:, 0].min(), lm[:, 0].max()
                miny, maxy = lm[:, 1].min(), lm[:, 1].max()
                cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                bw, bh = (maxx - minx), (maxy - miny)
                expand = 1.2
                half_w, half_h = (bw * expand) / 2.0, (bh * expand) / 2.0
                x1, x2 = int(max(0, cx - half_w)), int(min(frm.shape[1], cx + half_w))
                y1, y2 = int(max(0, cy - half_h)), int(min(frm.shape[0], cy + half_h))
                crop = cv2.resize(frm[y1:y2, x1:x2], FRAME_SIZE)
                rel_x = (lm[:, 0] - x1) / (x2 - x1 + 1e-8)
                rel_y = (lm[:, 1] - y1) / (y2 - y1 + 1e-8)
                rel = np.stack([rel_x, rel_y], axis=1).astype(np.float32)
                crops.append(crop)
                heatmaps.append(make_heatmap_from_landmarks(rel))
                lmarks.append(rel)

            tmp = str(cache_file) + ".tmp"
            with open(tmp, 'wb') as f:
                np.savez_compressed(
                    f,
                    modality="video",
                    video_path=str(file_path),
                    emotion=emotion,
                    fps=float(used_fps),
                    n_frames=len(frames),
                    crops=np.asarray(crops, dtype=np.uint8),
                    heatmaps=np.asarray(heatmaps, dtype=np.float32),
                    landmarks=np.asarray(lmarks, dtype=np.float32),
                )

            os.replace(tmp, cache_file)
            Path(done_file).touch()
            return str(cache_file), {}

        elif modality == "audio":
            info = sf.info(str(file_path))
            tmp = str(cache_file) + ".tmp"
            np.savez_compressed(
                tmp,
                modality="audio",
                audio_path=np.array([str(file_path)], dtype=object),
                emotion=emotion,
                duration=float(info.duration),
            )
            os.replace(tmp, cache_file)
            Path(done_file).touch()
            return str(cache_file), {}



    except Exception as e:
        print(f"[Stage A] ❌ Error {file_path}: {e}")
        return None, {'error': str(e)}

def stage_a_parallel(entries, cache_dir, n_workers):
    ensure_dir(cache_dir)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=MAXTASKSPERCHILD) as pool:
        results = list(tqdm(pool.imap_unordered(stage_a_worker, entries),
                            total=len(entries), desc="Stage A (CPU)"))
    return results

# ---------------- Stage B (GPU) ----------------
class FaceFeatureResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(2048, 512)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)

def compute_audio_mel_gpu(path, device):
    waveform, sr = torchaudio.load(path)
    if sr != AUDIO_SR:
        waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SR)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=AUDIO_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    ).to(device)
    mel = mel_spec(waveform)
    db = torchaudio.transforms.AmplitudeToDB().to(device)
    logmel = db(mel)
    return logmel.squeeze(0).cpu().numpy()

def stage_b_gpu_processing(caches, out_root: Path, device: torch.device, batch: int):
    model = FaceFeatureResNet50().to(device).eval()
    saved = []
    pbar = tqdm(total=len(caches), desc="Stage B (GPU)", ncols=100)

    for cf in caches:
        try:
            npz = np.load(cf, allow_pickle=True)
            mod = str(npz.get("modality", "unknown"))
            emo = str(npz.get("emotion", "unknown"))

            emo_dir = out_root / emo
            vid_dir = emo_dir / "video"
            aud_dir = emo_dir / "audio"
            ensure_dir(vid_dir)
            ensure_dir(aud_dir)

            if mod == "video":
                crops = npz["crops"].astype(np.float32) / 255.0
                fps = float(npz["fps"])
                vname = Path(npz["video_path"]).stem
                all_feats = []
                for i in range(0, len(crops), batch):
                    batch_imgs = torch.tensor(crops[i:i+batch]).permute(0, 3, 1, 2).to(device)
                    with torch.no_grad():
                        feats = model(batch_imgs)
                    all_feats.append(feats.cpu().numpy())
                feats_all = np.concatenate(all_feats, axis=0)
                np.savez_compressed(vid_dir / f"{vname}_video_feats.npz",
                                    features=feats_all, fps=fps, emotion=emo)
                saved.append(vname)

            elif mod == "audio":
                apath = npz["audio_path"].item() if isinstance(npz["audio_path"], np.ndarray) else str(npz["audio_path"])
                mel = compute_audio_mel_gpu(apath, device)
                aname = Path(apath).stem
                np.savez_compressed(aud_dir / f"{aname}_audio_feats.npz",
                                    mel=mel, emotion=emo)
                saved.append(aname)

        except Exception as e:
            print(f"[Stage B] Error on {cf}: {e}")
        finally:
            pbar.update(1)

    pbar.close()
    print(f"✅ Stage B complete — {len(saved)} features saved.")
    return saved

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_root', type=str, default='data_sorted')
    ap.add_argument('--index_csv', type=str, default='data_sorted/dataset_index.csv')
    ap.add_argument('--output_root', type=str, default='features')
    ap.add_argument('--cache_dir', type=str, default='cache')
    ap.add_argument('--preview', type=int, default=0)
    ap.add_argument('--run_stage', type=str, default='both', choices=['A', 'B', 'both'])
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_workers = get_cpu_worker_count()
    print(f"Using {n_workers} CPU workers | device={device}")

    input_root, out_root, cache = Path(args.input_root), Path(args.output_root), Path(args.cache_dir)
    ensure_dir(out_root)
    ensure_dir(cache)

    if args.force:
        print(f"⚠️ Clearing cache {cache} ...")
        for f in cache.glob("*"):
            try:
                f.unlink()
            except Exception as e:
                print(f"  Cannot remove {f}: {e}")

    df = pd.read_csv(args.index_csv)
    print(f"✅ Loaded index.csv with {len(df)} entries")

    entries = []
    for _, row in df.iterrows():
        emo = row.get('emotion')
        vid, aud = row.get('video_path'), row.get('audio_path')
        if pd.notna(vid):
            vpath = Path(vid)
            if not vpath.exists():
                vpath = Path(input_root) / vid
            if vpath.exists() and vpath.suffix.lower() == ".mp4":
                entries.append((str(vpath), 'video', emo, str(cache)))
        if pd.notna(aud):
            apath = Path(aud)
            if not apath.exists():
                apath = Path(input_root) / aud
            if apath.exists():
                entries.append((str(apath), 'audio', emo, str(cache)))

    if args.preview > 0:
        entries = random.sample(entries, min(args.preview, len(entries)))
        print(f"🎞️ Preview: {len(entries)} samples")

    if not entries:
        print("❌ No valid entries found")
        return

    valid = []
    if args.run_stage in ['A', 'both']:
        res = stage_a_parallel(entries, cache, n_workers)
        valid = [r[0] for r in res if r and r[0]]

    if args.run_stage in ['B', 'both']:
        if not valid:
            valid = sorted(str(p) for p in Path(cache).glob("*.npz"))
        if valid:
            stage_b_gpu_processing(valid, out_root, device, auto_batch_size_by_vram())

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
