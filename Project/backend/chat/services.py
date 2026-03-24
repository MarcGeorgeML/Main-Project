from pathlib import Path
import shutil
import subprocess
import uuid
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from ..entities.chat import Chat


TEMP_DIR = Path(__file__).resolve().parent.parent.parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)


# -----------------------------
# Video Handling
# -----------------------------
def save_video(video: UploadFile, user_id: str, request_id: str) -> Path:
    """
    Save uploaded video as a properly encoded MP4 (H264/AAC).

    Browser recordings arrive as video/webm (VP8+Opus). If we just rename
    them to .mp4 the container label lies and the inference pipeline's frame /
    audio extractors crash with 'tuple index out of range'.  Transcoding every
    file through ffmpeg guarantees the engine always receives a genuine MP4
    regardless of the source.
    """
    raw_path = TEMP_DIR / f"{user_id}-{request_id}.raw"
    mp4_path = TEMP_DIR / f"{user_id}-{request_id}.mp4"

    # ── 1. Write raw bytes to disk ───────────────────────────────────────────
    video.file.seek(0)
    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # ── 2. Transcode to H264/AAC MP4 ────────────────────────────────────────
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

    # ── 3. Always remove the raw file ───────────────────────────────────────
    raw_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcoding failed: {result.stderr}")

    return mp4_path


# -----------------------------
# DB Operations
# -----------------------------
def get_latest_emotion(db: Session, user_id: uuid.UUID) -> str:
    """Get latest emotional state of the user."""
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
    """Delete all chats for a user."""
    stmt = delete(Chat).where(Chat.user_id == user_id)
    db.execute(stmt)
    db.commit()


def get_all_chats(db: Session, user_id: uuid.UUID) -> List[Chat]:
    """Return chat history ordered oldest-first."""
    stmt = (
        select(Chat)
        .where(Chat.user_id == user_id)
        .order_by(Chat.created_at.asc())
    )
    return db.execute(stmt).scalars().all()