from pydantic import BaseModel
from typing import Optional, List


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class ConversationUpdate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    message: str


class ChatRequest(BaseModel):
    message: str


class ChatHistoryItem(BaseModel):
    """A single previous turn passed to the engine for LLM context."""
    user_message: str
    ai_response: str
    emotion: str
    confidence: float


class JobMsgType(BaseModel):
    user_id: str
    type: str
    data: str
    latest_emotion: str
    history: List[ChatHistoryItem] = []
