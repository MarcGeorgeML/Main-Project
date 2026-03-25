from pydantic import BaseModel
from typing import List, Optional

# {
#     'request_id': '4b1268e7-e2b2-4d74-9684-89d84a1453a3', 
#     'data': {
#         "user_id": "779d5f22-ef1a-46fc-ad0f-c1fe9ec470ba", 
#         "type": "text", 
#         "message": "hello"
#     }
# }




class ChatHistoryItem(BaseModel):
    """A single previous turn passed from the backend for LLM context."""
    user_message: str
    ai_response: str
    emotion: str
    confidence: float


class JobData(BaseModel):
    user_id: str
    type: str
    data: str
    latest_emotion: str
    history: List[ChatHistoryItem] = []


class RequestType(BaseModel):
    request_id: str
    data: JobData