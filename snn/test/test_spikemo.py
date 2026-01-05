import os
import sys
import torch
import torch.nn.functional as F
import librosa
import numpy as np

# --------------------------------------------------
# Ensure current directory is in PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from SpikEmo_Model import SpikEmo
from spikformer import Spikformer

# --------------------------------------------------
# CONFIG (match training as closely as possible)
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "IEMOCAP"
N_CLASSES = 6
N_SPEAKERS = 2

ROBERTA_DIM = 768          # text feature dim (unused but required)
D_M_AUDIO = 40             # must match training
D_M_VISUAL = 512           # dummy visual dim
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
# Audio feature extraction
# --------------------------------------------------
def extract_audio(mp3_path, target_dim=D_M_AUDIO):
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=target_dim,
        n_fft=512, hop_length=256
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.T.astype(np.float32)  # (L, D)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    checkpoint = "model/spikemo_best_IEMOCAP.pt"
    audio_file = "audio/cry.mp3"

    print("Device:", DEVICE)

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
    # Build SpikEmo
    # -----------------------------
    model = SpikEmo(
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

    # -----------------------------
    # Load weights
    # -----------------------------
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    print("Model loaded successfully")

    # -----------------------------
    # Prepare input
    # -----------------------------
    audio_feats = extract_audio(audio_file)
    L = audio_feats.shape[0]

    texts = torch.zeros((1, L, ROBERTA_DIM), device=DEVICE)
    audios = torch.tensor(audio_feats).unsqueeze(0).to(DEVICE)
    visuals = torch.zeros((1, L, D_M_VISUAL), device=DEVICE)

    speaker_masks = torch.zeros((1, L, 2), device=DEVICE)
    utterance_masks = torch.ones((1, L), device=DEVICE)
    padded_labels = torch.zeros((1, L), device=DEVICE).long()

    # -----------------------------
    # Forward pass
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

    # -----------------------------
    # Output
    # -----------------------------
    print("\nEmotion probabilities:")
    for i, p in enumerate(mean_probs):
        print(f"{LABEL_MAP[i]:12s}: {p.item():.4f}")

    pred = mean_probs.argmax().item()
    print(f"\nPredicted emotion: {LABEL_MAP[pred]}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
