import torch
import soundfile as sf
import os

def extract_audio_features(audio_ref, smile, device):
    if not os.path.splitext(audio_ref)[1]:
        audio_ref += ".mp3"

    audio, sr = sf.read(audio_ref)
    feats = smile.process_signal(audio, sr)

    tensor = torch.tensor(
        feats.values, dtype=torch.float32, device=device
    )
    return tensor.unsqueeze(0)
