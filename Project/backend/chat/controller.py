# chat/router.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import uuid

from ..middleware.auth_middleware import auth_middleware
from ..auth.models import TokenData
from ..data.database import DbSession
from ..data.redis_client import redis_client
from . import services
from .models import JobMsgType

chat_router = APIRouter(prefix="/chats", tags=["Chat"])


# ── Sessions ──────────────────────────────────────────────────────────────────

@chat_router.post("/sessions")
async def create_session(
    db: DbSession,
    payload: TokenData = Depends(auth_middleware),
):
    session = services.create_session(db, payload.user_id)
    return {"session_id": str(session.id), "created_at": session.created_at}


@chat_router.get("/sessions")
async def get_sessions(
    db: DbSession,
    payload: TokenData = Depends(auth_middleware),
):
    sessions = services.get_all_sessions(db, payload.user_id)
    result = []
    for s in sessions:
        first_msg = s.chats[0].transcription if s.chats else None
        result.append({
            "session_id": str(s.id),
            "first_message": first_msg,
            "updated_at": s.updated_at,
            "created_at": s.created_at,
        })
    return result


@chat_router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: uuid.UUID,
    db: DbSession,
    payload: TokenData = Depends(auth_middleware),
):
    session = services.get_session(db, session_id, payload.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    services.delete_session(db, session_id, payload.user_id)
    return {"message": "Session deleted"}


# ── Video / Chat ──────────────────────────────────────────────────────────────

@chat_router.post("/sessions/{session_id}/video")
async def send_video(
    session_id: uuid.UUID,
    db: DbSession,
    video: UploadFile = File(...),
    payload: TokenData = Depends(auth_middleware),
):
    session = services.get_session(db, session_id, payload.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        request_id = str(uuid.uuid4())
        video_path = services.save_video(
            video=video,
            user_id=str(payload.user_id),
            request_id=request_id,
        )

        latest_emotion = services.get_latest_emotion(db, payload.user_id)
        job_data = JobMsgType(
            user_id=str(payload.user_id),
            type="video",
            data=str(video_path),
            latest_emotion=latest_emotion,
        )

        await redis_client.send_to_engine(request_id, job_data)
        response = await redis_client.wait_for_response(request_id)

        chat = services.create_chat_entry(
            db=db,
            user_id=payload.user_id,
            session_id=session_id,
            video_url=str(video_path),
            transcription=response.get("transcription"),
            detected_emotion=response.get("emotion"),
            emotion_confidence=response.get("confidence"),
            ai_response=response.get("message"),
        )

        services.touch_session(db, session_id)

        return {
            "chat_id": str(chat.id),
            "transcription": chat.transcription,
            "ai_response": chat.ai_response,
            "emotion": chat.detected_emotion,
            "confidence": chat.emotion_confidence,
            "latest_emotional_state": chat.latest_emotional_state,
            "created_at": chat.created_at,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/sessions/{session_id}")
async def get_session_history(
    session_id: uuid.UUID,
    db: DbSession,
    payload: TokenData = Depends(auth_middleware),
):
    session = services.get_session(db, session_id, payload.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    chats = services.get_chats_by_session(db, session_id, payload.user_id)
    return [
        {
            "id": str(chat.id),
            "video_url": chat.video_url,
            "transcription": chat.transcription,
            "ai_response": chat.ai_response,
            "emotion": chat.detected_emotion,
            "confidence": chat.emotion_confidence,
            "created_at": chat.created_at,
        }
        for chat in chats
    ]


# ── Kept for compatibility ─────────────────────────────────────────────────────

@chat_router.delete("")
async def delete_all_history(
    db: DbSession,
    payload: TokenData = Depends(auth_middleware),
):
    services.delete_all_chats(db, payload.user_id)
    return {"message": "Chat history cleared"}


@chat_router.get("/emotion")
async def get_current_emotion(
    db: DbSession,
    payload: TokenData = Depends(auth_middleware),
):
    emotion = services.get_latest_emotion(db, payload.user_id)
    return {"current_emotion": emotion}