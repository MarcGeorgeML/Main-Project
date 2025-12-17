from sqlalchemy import Column, String, ForeignKey, DateTime, Enum, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
import enum
from ..data.database import Base


class SenderType(enum.Enum):
    USER = "USER"
    AI = "AI"


class MessageType(enum.Enum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    sender = Column(Enum(SenderType), nullable=False)
    message_type = Column(Enum(MessageType), nullable=False)
    message = Column(String, nullable=False)
    transcription = Column(Text, nullable=True)  # Added field for audio transcriptions
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    conversation = relationship("Conversation", back_populates="messages")