from fastapi import HTTPException, UploadFile
from pathlib import Path
import shutil
import subprocess
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from fastapi import UploadFile
import uuid
from typing import Optional, List
from ..data.database import DbSession
from ..entities.conversation import Conversation
from ..entities.message import Message, SenderType, MessageType
from sqlalchemy import func
import tempfile
from minio.error import S3Error
from ..data.minio_client import minio_client, ensure_bucket
import os



def create_conversation(
    db: DbSession,
    user_id: uuid.UUID
) -> Conversation:
    """
    Create a new conversation with auto-incremented title:
    Chat1, Chat2, Chat3, ...
    (per user)
    """

    # Count existing conversations for this user
    count_stmt = select(func.count(Conversation.id)).where(
        Conversation.user_id == user_id
    )
    conversation_count = db.execute(count_stmt).scalar() or 0

    # Generate title
    title = f"Chat{conversation_count + 1}"

    conversation = Conversation(
        user_id=user_id,
        title=title
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

def get_all_conversations(
    db: DbSession,
    user_id: uuid.UUID
) -> List[Conversation]:
    """
    Get all conversations for a user (latest first)
    """
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
    )

    result = db.execute(stmt)
    conversations = result.scalars().all()

    return conversations


def delete_conversation(db: DbSession, conversation_id: str, user_id: uuid.UUID) -> bool:
    """Delete a conversation (only if it belongs to the user)"""
    conversation = db.query(Conversation).filter(
        Conversation.id == uuid.UUID(conversation_id),
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        return False
    
    db.delete(conversation)
    db.commit()
    return True


def update_conversation_title(
    db: DbSession,
    conversation_id: str,
    user_id: uuid.UUID,
    title: str
) -> Optional[Conversation]:
    """Update the title of a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == uuid.UUID(conversation_id),
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        return None
    
    conversation.title = title
    db.commit()
    db.refresh(conversation)
    return conversation


def get_conversation_messages(
    db: DbSession,
    conversation_id: str,
    user_id: uuid.UUID
) -> Optional[List[Message]]:
    """Get all messages from a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == uuid.UUID(conversation_id),
        Conversation.user_id == user_id
    ).options(selectinload(Conversation.messages)).first()
    
    if not conversation:
        return None
    
    return conversation.messages


def add_message_to_conversation(
    db: DbSession,
    conversation_id: str,
    user_id: uuid.UUID,
    sender: str,
    message_type: str,
    message: str,
    transcription: Optional[str] = None  # Added transcription parameter
) -> Optional[Message]:
    """Add a message to a conversation"""
    # Verify conversation belongs to user
    conversation = db.query(Conversation).filter(
        Conversation.id == uuid.UUID(conversation_id),
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        return None
    
    # Create message
    new_message = Message(
        conversation_id=uuid.UUID(conversation_id),
        sender=SenderType[sender],
        message_type=MessageType[message_type],
        message=message,
        transcription=transcription  # Added transcription field
    )
    
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    return new_message




def save_and_convert_audio(
    audio: UploadFile,
    user_id: str,
    request_id: str
) -> str:
    """
    Converts uploaded audio to MP3 and uploads to MinIO.
    Returns MinIO object name.
    """

    ensure_bucket()

    # temp input file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as tmp_in:
        audio.file.seek(0)
        tmp_in.write(audio.file.read())
        input_path = tmp_in.name

    # temp output mp3
    output_path = f"{input_path}.mp3"

    # ffmpeg conversion
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, output_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"FFmpeg failed: {result.stderr}"
        )

    # MinIO object path
    object_name = f"{user_id}/{request_id}.mp3"

    try:
        minio_client.fput_object(
            bucket_name=os.getenv("MINIO_BUCKET", "audio"),
            object_name=object_name,
            file_path=output_path,
            content_type="audio/mpeg"
        )
    except S3Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup temp files
        os.remove(input_path)
        os.remove(output_path)

    return object_name