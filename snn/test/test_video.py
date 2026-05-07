import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import cv2
# from moviepy.editor import VideoFileClip
from torchvision import models, transforms
import subprocess
import tempfile
import os
import librosa
import numpy as np

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from SentiCore_Model import SentiCore
from spikformer import Spikformer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "IEMOCAP"
N_CLASSES = 6
N_SPEAKERS = 2

ROBERTA_DIM = 768
D_M_AUDIO = 40
D_M_VISUAL = 512
MODEL_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8

D_G = 64
D_P = 64
D_E = 64
D_H = 64
D_A = 32

DROPOUT = 0.1
DROPOUT_REC = 0.1

TAU = 10.0
COMMON_THR = 1.0
SPIKE_T = 32

MULTI_ATTN = True
LISTENER_STATE = True
CONTEXT_ATTENTION = True

LABEL_MAP = {
    0: "happiness",
    1: "sadness",
    2: "neutral",
    3: "anger",
    4: "excitement",
    5: "frustration"
}

# --------------------------------------------------
# AUDIO EXTRACTION FROM VIDEO
# --------------------------------------------------
def extract_audio_from_video(video_path, target_dim=40):
    """
    Robust audio extraction using ffmpeg + librosa
    Returns: (L, 40)
    """

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    # Extract mono 16kHz audio using ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    # Load audio
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    os.remove(wav_path)

    if y.size == 0:
        raise RuntimeError("Extracted audio is empty")

    # Mel features (MATCH TRAINING DIM)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=target_dim,
        n_fft=512,
        hop_length=256
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    return mel.T.astype(np.float32)  # (L, 40)

# --------------------------------------------------
# VISUAL FEATURE EXTRACTION
# --------------------------------------------------
def extract_visual_features(video_path, target_len):
    cap = cv2.VideoCapture(video_path)

    backbone = models.resnet18(pretrained=True)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(DEVICE).eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    features = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError("Could not read video frames (frame_count=0)")

    # sample evenly across video
    frame_indices = np.linspace(0, frame_count - 1, target_len).astype(int)

    current_idx = 0
    selected = set(frame_indices.tolist())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in selected:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess(frame).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = backbone(frame).squeeze().cpu().numpy()

            features.append(feat)

        current_idx += 1

    cap.release()

    if len(features) == 0:
        raise RuntimeError("No visual features extracted — check video file")

    features = np.stack(features)

    # pad or trim to exact length
    if features.shape[0] < target_len:
        pad = np.repeat(features[-1][None, :], target_len - features.shape[0], axis=0)
        features = np.concatenate([features, pad], axis=0)
    else:
        features = features[:target_len]

    return features.astype(np.float32)  # (L, 512)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    checkpoint = "model/SentiCore_best_IEMOCAP.pt"
    video_file = "video/test3.mp4"

    # -----------------------------
    # Build Spikformer
    # -----------------------------
    spikformer = Spikformer(
        depths=NUM_LAYERS,
        tau=TAU,
        common_thr=COMMON_THR,
        dim=MODEL_DIM,
        T=SPIKE_T,
        heads=NUM_HEADS
    ).to(DEVICE)

    # -----------------------------
    # Build SentiCore
    # -----------------------------
    model = SentiCore(
        dataset=DATASET,
        multi_attn_flag=MULTI_ATTN,
        roberta_dim=ROBERTA_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        D_m_audio=D_M_AUDIO,
        D_m_visual=D_M_VISUAL,
        D_g=D_G,
        D_p=D_P,
        D_e=D_E,
        D_h=D_H,
        n_classes=N_CLASSES,
        n_speakers=N_SPEAKERS,
        listener_state=LISTENER_STATE,
        context_attention=CONTEXT_ATTENTION,
        D_a=D_A,
        dropout_rec=DROPOUT_REC,
        device=DEVICE,
        spikformer_model=spikformer
    ).to(DEVICE)

    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    # -----------------------------
    # FEATURE EXTRACTION
    # -----------------------------
    audio_feats = extract_audio_from_video(video_file)
    L = audio_feats.shape[0]

    visual_feats = extract_visual_features(video_file, L)

    texts = torch.zeros((1, L, ROBERTA_DIM), device=DEVICE)
    audios = torch.tensor(audio_feats).unsqueeze(0).to(DEVICE)
    visuals = torch.tensor(visual_feats).unsqueeze(0).to(DEVICE)

    speaker_masks = torch.zeros((1, L, 2), device=DEVICE)
    utterance_masks = torch.ones((1, L), device=DEVICE)
    padded_labels = torch.zeros((1, L), device=DEVICE).long()

    # -----------------------------
    # INFERENCE
    # -----------------------------
    with torch.no_grad():
        _, _, _, _, logits = model(
            texts,
            audios,
            visuals,
            speaker_masks,
            utterance_masks,
            padded_labels
        )

        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)

    print("\nEmotion probabilities:")
    for i, p in enumerate(mean_probs):
        print(f"{LABEL_MAP[i]:12s}: {p.item():.4f}")

    pred = mean_probs.argmax().item()
    print(f"\nPredicted emotion: {LABEL_MAP[pred]}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
