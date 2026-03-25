"""
emotion_state.py
----------------
Computes a global emotional state string from a list of recent chat turns,
each of which carries a detected emotion and a confidence score.

The strategy:
  Uses the LLM to analyse the emotional arc of the conversation and return
  a single best-fit emotion, strongly prioritising the latest message.
  The output is always one of the 6 canonical emotions.
"""

from typing import List, Dict, Optional
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# The 6 canonical emotions the global state can be.
CANONICAL_EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad"}

# Emotions the SNN model can return – keep in sync with your model.
KNOWN_EMOTIONS = {
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "anxious"
}

# Mapping of non-canonical emotions to closest canonical equivalent
EMOTION_FALLBACK_MAP = {
    "surprise": "neutral",
    "anxious":  "fear",
}


def _normalise_emotion(emotion: str) -> str:
    """Map any known emotion to its canonical form."""
    e = emotion.lower().strip()
    if e in CANONICAL_EMOTIONS:
        return e
    return EMOTION_FALLBACK_MAP.get(e, "neutral")


def compute_emotion_state(
    history: List,
    llm: Optional[ChatGroq] = None,
) -> str:
    """
    Parameters
    ----------
    history : list of objects/dicts with attributes:
        - emotion    : str   – detected emotion for that turn
        - confidence : float – model confidence (0-1)
    llm     : ChatGroq instance (optional). If None, falls back to
              weighted heuristic.

    Returns
    -------
    str  – exactly one of: "angry", "disgust", "fear", "happy", "neutral", "sad"
    """
    if not history:
        return "neutral"

    # ── Build a compact summary of the emotional arc ─────────────────────────
    arc_lines = []
    for i, turn in enumerate(history, 1):
        emotion    = (getattr(turn, "emotion",    None) or "neutral").lower().strip()
        confidence = float(getattr(turn, "confidence", None) or 0.5)
        if emotion not in KNOWN_EMOTIONS:
            emotion = "neutral"
        arc_lines.append(f"  Turn {i}: emotion={emotion}, confidence={confidence:.2f}")

    arc_text = "\n".join(arc_lines)

    # The latest turn (last in list) is given to the LLM explicitly.
    latest = history[-1]
    latest_emotion    = (getattr(latest, "emotion",    None) or "neutral").lower().strip()
    latest_confidence = float(getattr(latest, "confidence", None) or 0.5)
    if latest_emotion not in KNOWN_EMOTIONS:
        latest_emotion = "neutral"

    # ── LLM path ─────────────────────────────────────────────────────────────
    if llm is not None:
        system_msg = SystemMessage(content=(
            "You are an emotion analysis assistant. "
            "Given a sequence of detected emotions from a therapy session, "
            "you must decide the single overall emotional state of the user. "
            "You MUST return exactly one word from this list: "
            "angry, disgust, fear, happy, neutral, sad. "
            "No punctuation, no explanation, no other words — just the single emotion word. "
            "Heavily weight the latest turn's emotion when making your decision."
        ))

        human_msg = HumanMessage(content=(
            f"Emotional arc of the session (oldest to newest):\n{arc_text}\n\n"
            f"LATEST turn (most important): emotion={latest_emotion}, "
            f"confidence={latest_confidence:.2f}\n\n"
            "What is the single overall emotional state? "
            "Reply with exactly one word from: angry, disgust, fear, happy, neutral, sad."
        ))

        try:
            response = llm.invoke([system_msg, human_msg])
            result   = response.content.strip().lower().split()[0]
            if result in CANONICAL_EMOTIONS:
                return result
            # If LLM returned something unexpected, fall through to heuristic
        except Exception as e:
            print(f"[emotion_state] LLM call failed, using heuristic fallback: {e}")

    # ── Heuristic fallback (no LLM or LLM returned unexpected value) ─────────
    # Weighted accumulation, with the latest turn's weight doubled.
    from collections import defaultdict
    scores: Dict[str, float] = defaultdict(float)

    for i, turn in enumerate(history):
        emotion    = (getattr(turn, "emotion",    None) or "neutral").lower().strip()
        confidence = float(getattr(turn, "confidence", None) or 0.5)
        canonical  = _normalise_emotion(emotion)

        # Double the weight for the latest turn
        multiplier = 2.0 if i == len(history) - 1 else 1.0
        scores[canonical] += confidence * multiplier

    best = max(scores, key=lambda e: scores[e])
    return best