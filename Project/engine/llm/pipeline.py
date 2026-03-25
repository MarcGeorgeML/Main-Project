"""
pipeline.py
-----------
LangChain + Groq chat pipeline for the empathetic therapist assistant.

Accepts:
  - transcription      : what the user just said
  - current_emotion    : emotion detected in the current turn
  - current_confidence : confidence score for current emotion
  - history            : list of the last N turns (each has user_message,
                         ai_response, emotion, confidence)
  - emotion_state      : pre-computed global emotional state string

Returns:
  str  – the AI therapist's response
"""

import os
from typing import List, Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are SentiCore, a warm, compassionate AI therapist and emotional support companion. \
Your role is to help users process and understand their emotions, feel genuinely heard, \
and find a path toward emotional well-being.

You have access to:
1. What the user said (their transcription).
2. The emotion detected by an AI model from their voice/face, along with a confidence score (0–1).
3. The last few messages of the conversation with their detected emotions.
4. A global emotional state summarising the overall emotional journey so far.

Guidelines for your responses:
- Always acknowledge the user's current emotion directly but gently. \
  If the detected emotion is high-confidence (≥ 0.70), trust it strongly; \
  if low-confidence (< 0.50), treat it as a soft signal and focus more on their words.
- Be empathetic, never clinical. Speak like a caring friend who also understands psychology.
- Encourage the user to open up gradually — ask ONE thoughtful follow-up question per turn. \
  Do NOT bombard them with multiple questions.
- If the user is happy or excited, celebrate that with them and explore what's going well.
- If the user is sad, anxious, or fearful, validate their feelings and gently explore the root cause.
- If the user is angry, acknowledge their frustration without judgment and help them feel understood.
- If the user appears neutral or surprised, stay curious and inviting.
- Never diagnose, never give medical advice, never tell the user how they "should" feel.
- Keep responses concise (3–5 sentences) unless the user is clearly in distress and needs more.
- Use the conversation history to maintain continuity — reference what they shared earlier when relevant.
- The global emotional state gives you the broader picture of this session; \
  use it to gauge whether the user is improving, staying the same, or escalating.

"""

# ---------------------------------------------------------------------------
# History formatting helper
# ---------------------------------------------------------------------------

def _format_history_context(history: List[Dict]) -> str:
    """
    Builds a compact context block appended to the human turn so the model
    understands the emotional arc of the conversation without it inflating
    the system prompt.
    """
    if not history:
        return ""

    lines = ["[Conversation history – most recent last]"]
    for i, turn in enumerate(history, 1):
        emotion = getattr(turn, "emotion", "unknown")
        conf = getattr(turn, "confidence", 0.0)
        user_msg = getattr(turn, "user_message", "")
        ai_msg = getattr(turn, "ai_response", "")
        lines.append(
            f"Turn {i}: User said: \"{user_msg}\" "
            f"(detected emotion: {emotion}, confidence: {conf:.2f})\n"
            f"         You replied: \"{ai_msg}\""
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class LLMChatPipeline:
    """
    Stateless pipeline — pass full history on every call.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/"
            )

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=api_key,
        )

    # ------------------------------------------------------------------

    def generate_response(
        self,
        transcription: str,
        current_emotion: str,
        current_confidence: float,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Parameters
        ----------
        transcription      : str   – user's spoken text this turn
        current_emotion    : str   – emotion detected this turn
        current_confidence : float – confidence for current emotion (0-1)
        emotion_state      : str   – global emotion state for this session
        history            : list  – previous turns (dicts with keys:
                                     user_message, ai_response, emotion, confidence)

        Returns
        -------
        str – therapist's response
        """
        history = history or []

        # Build the system message with dynamic emotion_state injected
        system_content = SYSTEM_PROMPT

        # Build LangChain message list
        messages: List = [SystemMessage(content=system_content)]

        # Inject summarised history as an AI/Human pair so the model has
        # the full conversational arc in its context window.
        history_context = _format_history_context(history)
        if history_context:
            messages.append(
                HumanMessage(content="[Context from our conversation so far]")
            )
            messages.append(AIMessage(content=history_context))

        # Build the current human turn with emotion metadata
        confidence_label = (
            "high" if current_confidence >= 0.70
            else "medium" if current_confidence >= 0.50
            else "low"
        )

        current_turn = (
            f"{transcription}\n\n"
            f"[Detected emotion: {current_emotion} "
            f"(confidence: {current_confidence:.2f} – {confidence_label})]"
        )
        messages.append(HumanMessage(content=current_turn))

        # Call the LLM
        response = self.llm.invoke(messages)
        return response.content.strip()