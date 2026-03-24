import re
from faster_whisper import WhisperModel


class WhisperSegmenter:
    def __init__(self, model_size="base", device="cuda"):
        self.model = WhisperModel(model_size, device=device)
        self.punctuation_pattern = re.compile(r"[.!?]")

    def segment_audio(self, audio_path):
        segments, _ = self.model.transcribe(
            audio_path, beam_size=5, word_timestamps=True, vad_filter=True
        )

        utterances = []
        for segment in segments:
            text = segment.text.strip()

            if len(text) == 0:
                continue

            duration = segment.end - segment.start

            if duration < 0.5 or duration > 10:
                continue

            utterances.append(
                {"start": float(segment.start), "end": float(segment.end), "text": text}
            )
        return utterances
