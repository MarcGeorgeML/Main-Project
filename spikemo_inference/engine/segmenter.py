from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(r"c:\Users\Marc\Desktop\Programming\Research\spikemo_inference\engine\.env"))

from faster_whisper import WhisperModel
from .audio_from_video import load_audio_from_video
import torch
import re


class UtteranceSegmenter:

    def __init__(self, model_size="base"):

        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

        # punctuation markers
        self.punct_pattern = re.compile(r"[.!?,]")

    def segment(self, video_path):

        audio, _ = load_audio_from_video(video_path)

        segments, _ = self.model.transcribe(
            audio,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )

        utterances = []

        for seg in segments:

            words = seg.words
            if not words:
                continue

            current_words = []
            start_time = words[0].start

            for w in words:

                current_words.append(w.word)

                # check punctuation
                if self.punct_pattern.search(w.word):

                    utterances.append({
                        "start": start_time,
                        "end": w.end,
                        "text": "".join(current_words).strip()
                    })

                    current_words = []
                    start_time = w.end

            # leftover words
            if current_words:

                utterances.append({
                    "start": start_time,
                    "end": words[-1].end,
                    "text": "".join(current_words).strip()
                })

        return utterances