from pathlib import Path
import shutil
import subprocess
import uuid
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from sqlalchemy import select, delete  # ← add `delete` here
from sqlalchemy.orm import Session

from ..entities.chat import Chat


TEMP_DIR = Path(__file__).resolve().parent.parent.parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)


# -----------------------------
# Video Handling
# -----------------------------
def save_video(video: UploadFile, user_id: str, request_id: str) -> Path:
    """Save uploaded video as MP4"""
    file_path = TEMP_DIR / f"{user_id}-{request_id}.mp4"

    video.file.seek(0)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    return file_path


# -----------------------------
# DB Operations
# -----------------------------
def get_latest_emotion(db: Session, user_id: uuid.UUID) -> str:
    """Get latest emotional state of the user"""
    stmt = (
        select(Chat.latest_emotional_state)
        .where(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .limit(1)
    )

    result = db.execute(stmt).scalar()

    return result if result else "neutral"


def create_chat_entry(
    db: Session,
    user_id: uuid.UUID,
    video_url: str,
    transcription: str,
    detected_emotion: str,
    emotion_confidence: float,
    ai_response: str,             
) -> Chat:
    chat = Chat(
        user_id=user_id,
        video_url=video_url,
        transcription=transcription,
        detected_emotion=detected_emotion,
        emotion_confidence=emotion_confidence,
        latest_emotional_state=detected_emotion,
        ai_response=ai_response,  
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

def delete_all_chats(db: Session, user_id: uuid.UUID) -> None:
    """Delete all chats for a user"""
    stmt = delete(Chat).where(Chat.user_id == user_id)
    db.execute(stmt)
    db.commit()


def get_all_chats(db: Session, user_id: uuid.UUID) -> List[Chat]:
    """Return chat history"""
    stmt = (
        select(Chat)
        .where(Chat.user_id == user_id)
        .order_by(Chat.created_at.asc())
    )
    return db.execute(stmt).scalars().all()

