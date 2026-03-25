"""
emotion_state.py
----------------
Computes a global emotional state string from a list of recent chat turns,
each of which carries a detected emotion and a confidence score.

The strategy:
  1. Accumulate a weighted score per emotion across all turns
     (weight = confidence, so high-confidence readings matter more).
  2. Filter to emotions whose accumulated weight exceeds a minimum threshold
     so that fleeting, low-confidence readings are ignored.
  3. Return the top emotions (up to MAX_LABELS) sorted by total weight,
     joined as a human-readable string like "sad and anxious".
"""

from typing import List, Dict
from collections import defaultdict

# Emotions the SNN model can return – keep in sync with your model.
KNOWN_EMOTIONS = {
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "anxious"
}

# Emotions that are essentially "no strong signal" – weight them lower.
NEUTRAL_EMOTIONS = {"neutral"}

MAX_LABELS = 2          # Maximum distinct emotions in the global label.
MIN_WEIGHT = 0.25       # Minimum accumulated weight to be included.
NEUTRAL_DISCOUNT = 0.5  # Multiply neutral confidence by this before accumulating.


def compute_emotion_state(history: List[Dict]) -> str:
    """
    Parameters
    ----------
    history : list of dicts with keys:
        - "emotion"    : str   – detected emotion for that turn
        - "confidence" : float – model confidence (0-1)

    Returns
    -------
    str  e.g. "sad", "sad and anxious", "happy", "neutral"
    """
    if not history:
        return "neutral"

    scores: Dict[str, float] = defaultdict(float)

    for turn in history:
        emotion = (getattr(turn, "emotion", None) or "neutral").lower().strip()
        confidence = float(getattr(turn, "confidence", None) or 0.5)

        if emotion not in KNOWN_EMOTIONS:
            emotion = "neutral"

        weight = confidence * (NEUTRAL_DISCOUNT if emotion in NEUTRAL_EMOTIONS else 1.0)
        scores[emotion] += weight

    # Filter out low-weight emotions
    significant = {e: w for e, w in scores.items() if w >= MIN_WEIGHT}

    if not significant:
        return "neutral"

    # Sort by weight descending, take top MAX_LABELS
    top = sorted(significant, key=lambda e: significant[e], reverse=True)[:MAX_LABELS]

    if len(top) == 1:
        return top[0]
    return " and ".join(top)