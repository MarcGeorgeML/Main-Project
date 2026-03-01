from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import uuid

from ..middleware.auth_middleware import auth_middleware
from ..auth.models import TokenData
from ..data.database import DbSession
from ..data.redis_client import redis_client
from . import services
from .models import JobMsgType

chat_router = APIRouter(prefix="/chats", tags=["Chat"])



@chat_router.post("/video")
async def send_video(
    db: DbSession,
    video: UploadFile = File(...),
    payload: TokenData = Depends(auth_middleware),
):
    try:
        request_id = str(uuid.uuid4())

        video_path = services.save_video(
            video=video,
            user_id=str(payload.user_id),
            request_id=request_id
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
            video_url=str(video_path),
            transcription=response.get("transcription"),
            detected_emotion=response.get("emotion"),
            emotion_confidence=response.get("confidence"),
            ai_response=response.get("message"),       
        )

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
    

@chat_router.get("")
async def get_chat_history(
    db: DbSession,
    payload: TokenData = Depends(auth_middleware),
):
    chats = services.get_all_chats(db, payload.user_id)
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

@chat_router.delete("")
async def delete_chat_history(
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


@chat_router.get("/test")
async def testing(payload: TokenData = Depends(auth_middleware)):
    print("/chats/test")
    request_id = str(uuid.uuid4())
    
    try:
        await redis_client.send_to_engine(
            request_id=request_id,
            data={"email": payload.email}
        )
        
        ack_response = await redis_client.wait_for_response(request_id, timeout=10.0)
        
        return {
            "message": ack_response.get("message"),
            "request_id": request_id
        }
    except TimeoutError:
        raise HTTPException(status_code=408, detail="Request Timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")