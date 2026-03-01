from pydantic import BaseModel

from pydantic import BaseModel
from typing import Optional


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class ConversationUpdate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    message: str

class ChatRequest(BaseModel):
    message: str

class JobMsgType(BaseModel):
    user_id: str
    type: str
    data: str
    latest_emotion: str
