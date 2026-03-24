from dataclasses import dataclass


EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
}


@dataclass
class UtterancePrediction:

    start: float
    end: float
    text: str
    emotion: str
    class_index: int
    confidence: float
    all_scores: dict

    def __repr__(self) -> str:
        return (
            f"[{self.start:.2f}s → {self.end:.2f}s] "
            f'"{self.text}" '
            f"→ {self.emotion.upper()} ({self.confidence * 100:.1f}%)"
        )