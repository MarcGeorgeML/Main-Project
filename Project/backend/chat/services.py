# chat/services.py
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import List, Optional

from fastapi import UploadFile
from sqlalchemy import select, delete, update
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from ..entities.chat import Chat, ChatSession


TEMP_DIR = Path(__file__).resolve().parent.parent.parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)


# -----------------------------
# Video Handling
# -----------------------------
def save_video(video: UploadFile, user_id: str, request_id: str) -> Path:
    raw_path = TEMP_DIR / f"{user_id}-{request_id}.raw"
    mp4_path = TEMP_DIR / f"{user_id}-{request_id}.mp4"

    video.file.seek(0)
    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(raw_path),
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-strict", "experimental",
            "-loglevel", "error",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
    )

    raw_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcoding failed: {result.stderr}")

    return mp4_path


# -----------------------------
# Session Operations
# -----------------------------
def create_session(db: Session, user_id: uuid.UUID) -> ChatSession:
    session = ChatSession(user_id=user_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_all_sessions(db: Session, user_id: uuid.UUID) -> List[ChatSession]:
    """Return all sessions for a user, ordered by most recently updated."""
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return db.execute(stmt).scalars().all()


def get_session(db: Session, session_id: uuid.UUID, user_id: uuid.UUID) -> Optional[ChatSession]:
    stmt = select(ChatSession).where(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id,
    )
    return db.execute(stmt).scalar_one_or_none()


def touch_session(db: Session, session_id: uuid.UUID) -> None:
    """Bump updated_at so session sorts to the top."""
    stmt = (
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=datetime.now(timezone.utc))
    )
    db.execute(stmt)
    db.commit()


def delete_session(db: Session, session_id: uuid.UUID, user_id: uuid.UUID) -> None:
    db.execute(delete(Chat).where(Chat.session_id == session_id))
    db.execute(
        delete(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    db.commit()


# -----------------------------
# DB Operations
# -----------------------------
def get_latest_emotion(db: Session, user_id: uuid.UUID) -> str:
    stmt = (
        select(Chat.latest_emotional_state)
        .where(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .limit(1)
    )
    result = db.execute(stmt).scalar()
    return result if result else "neutral"


def get_recent_chats_by_session(
    db: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    limit: int = 5,
) -> List[Chat]:
    """
    Return the most recent `limit` chats for a session, ordered oldest-first
    so they read as natural conversation history for the LLM.
    """
    # Fetch the N most recent rows (desc), then reverse for chronological order
    stmt = (
        select(Chat)
        .where(Chat.session_id == session_id, Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()
    return list(reversed(rows))


def create_chat_entry(
    db: Session,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    video_url: str,
    transcription: str,
    detected_emotion: str,
    emotion_confidence: float,
    ai_response: str,
) -> Chat:
    chat = Chat(
        user_id=user_id,
        session_id=session_id,
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
    db.execute(delete(Chat).where(Chat.user_id == user_id))
    db.execute(delete(ChatSession).where(ChatSession.user_id == user_id))
    db.commit()


def get_chats_by_session(db: Session, session_id: uuid.UUID, user_id: uuid.UUID) -> List[Chat]:
    stmt = (
        select(Chat)
        .where(Chat.session_id == session_id, Chat.user_id == user_id)
        .order_by(Chat.created_at.asc())
    )
    return db.execute(stmt).scalars().all()