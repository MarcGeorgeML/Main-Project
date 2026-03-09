from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    Path(r"c:\Users\Marc\Desktop\Programming\Research\spikemo_inference\engine\.env")
)

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from .audio_from_video import load_audio_from_video


class AudioFeatureExtractor:

    def __init__(self, device):

        self.device = device
        self.audio_cache = None
        self.sr = 16000

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        self.model.eval()
        if device.type == "cuda":
            self.model.half()

        self.proj = nn.Linear(768, 512).to(device)
        if device.type == "cuda":
            self.proj = self.proj.half()
        for p in self.proj.parameters():
            p.requires_grad = False

    def load_audio_segment(self, video_path, start, end):

        if self.audio_cache is None:
            audio, sr = load_audio_from_video(video_path, self.sr)
            self.audio_cache = audio

        s = int(start * self.sr)
        e = int(end * self.sr)

        segment = self.audio_cache[s:e]

        if len(segment) < 400:
            pad = np.zeros(400 - len(segment), dtype=np.float32)
            segment = np.concatenate([segment, pad])

        return segment.astype(np.float32)

    def extract_batch(self, audio_segments):

        inputs = self.processor.feature_extractor(
            audio_segments, sampling_rate=self.sr, return_tensors="pt", padding=True
        )

        input_values = inputs["input_values"].to(self.device)
        if self.device.type == "cuda":
            input_values = input_values.half()
        with torch.inference_mode():
            outputs = self.model(input_values)
        feats = outputs.last_hidden_state.mean(dim=1)
        feats = self.proj(feats)
        feats = feats.float()
        return feats
