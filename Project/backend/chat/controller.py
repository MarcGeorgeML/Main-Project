from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
import uuid

from ..middleware.auth_middleware import auth_middleware
from ..auth.models import TokenData
from ..data.database import DbSession
from ..data.redis_client import redis_client
from .models import ChatRequest, JobMsgType, ConversationCreate, ConversationUpdate
from . import services

chat_router = APIRouter(
    prefix='/chats',
    tags=["Chat"]
)


# Conversation endpoints
@chat_router.post("")
async def create_new_conversation(
    db: DbSession,
    payload: TokenData = Depends(auth_middleware)
):
    """
    Create a new conversation.
    Title is auto-generated as Chat1, Chat2, ...
    """
    try:
        new_conversation = services.create_conversation(
            db=db,
            user_id=payload.user_id
        )
        return {
            "id": str(new_conversation.id),
            "title": new_conversation.title,
            "created_at": new_conversation.created_at
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating conversation: {str(e)}"
        )

    

@chat_router.get("")
async def get_all_conversations(
    db: DbSession,
    payload: TokenData = Depends(auth_middleware)
):
    """Get all conversations for the user"""
    try:
        conversations = services.get_all_conversations(
            db=db,
            user_id=payload.user_id
        )
        return [
            {
                "id": str(conv.id),
                "title": conv.title,
                "created_at": conv.created_at
            }
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@chat_router.delete("/{chat_id}")
async def delete_conversation_endpoint(
    chat_id: str,
    db: DbSession,
    payload: TokenData = Depends(auth_middleware)
):
    """Delete a conversation"""
    try:
        success = services.delete_conversation(
            db=db,
            conversation_id=chat_id,
            user_id=payload.user_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"message": "Chat deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")


@chat_router.patch("/{chat_id}")
async def update_conversation_endpoint(
    chat_id: str,
    conversation_update: ConversationUpdate,
    db: DbSession,
    payload: TokenData = Depends(auth_middleware)
):
    """Update conversation title"""
    try:
        updated_conversation = services.update_conversation_title(
            db=db,
            conversation_id=chat_id,
            user_id=payload.user_id,
            title=conversation_update.title
        )
        if not updated_conversation:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {
            "id": str(updated_conversation.id),
            "title": updated_conversation.title
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating chat: {str(e)}")


@chat_router.get("/{chat_id}/messages")
async def get_messages(
    chat_id: str,
    db: DbSession,
    payload: TokenData = Depends(auth_middleware)
):
    """Get all messages from a conversation"""
    try:
        messages = services.get_conversation_messages(
            db=db,
            conversation_id=chat_id,
            user_id=payload.user_id
        )
        if messages is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return {
            "chat_id": chat_id,
            "messages": [
                {
                    "id": str(msg.id),
                    "sender": msg.sender.value,
                    "message_type": msg.message_type.value,
                    "message": msg.message,
                    "transcription": msg.transcription,  # Added transcription field
                    "created_at": msg.created_at
                }
                for msg in messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")


# Message endpoints
@chat_router.post("/{chat_id}/messages/text")
async def send_text_message(
    chat_id: str,
    req: ChatRequest,
    db: DbSession,
    payload: TokenData = Depends(auth_middleware)
):
    """Send a text message to a conversation"""
    try:
        # Save user message
        user_message = services.add_message_to_conversation(
            db=db,
            conversation_id=chat_id,
            user_id=payload.user_id,
            sender="USER",
            message_type="TEXT",
            message=req.message
        )
        
        if not user_message:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Process with AI engine
        request_id = str(uuid.uuid4())
        job_message = JobMsgType(
            user_id=payload.user_id,
            type="text",
            data=req.message
        )
        
        await redis_client.send_to_engine(request_id, job_message)
        response = await redis_client.wait_for_response(request_id)
        
        # Save AI response
        ai_message = services.add_message_to_conversation(
            db=db,
            conversation_id=chat_id,
            user_id=payload.user_id,
            sender="AI",
            message_type="TEXT",
            message=response.get("message")
        )
        
        return {
            "user_message": {
                "id": str(user_message.id),
                "message": user_message.message,
                "created_at": user_message.created_at
            },
            "ai_message": {
                "id": str(ai_message.id),
                "message": ai_message.message,
                "created_at": ai_message.created_at
            },
            "request_id": request_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")


@chat_router.post("/{chat_id}/messages/audio")
async def send_audio_message(
    db: DbSession,
    chat_id: str,
    audio: UploadFile = File(...),
    payload: TokenData = Depends(auth_middleware),
):
    """Send an audio message to a conversation"""
    try:
        request_id = str(uuid.uuid4())
        
        # Save and convert audio
        mp3_filename = services.save_and_convert_audio(
            audio=audio,
            user_id=payload.user_id,
            request_id=request_id
        )
        
        # Process with AI engine first to get transcription
        job_message = JobMsgType(
            user_id=payload.user_id,
            type="audio",
            data=mp3_filename.name
        )
        
        await redis_client.send_to_engine(request_id, job_message)
        response = await redis_client.wait_for_response(request_id)
        
        # Save user audio message with transcription
        user_message = services.add_message_to_conversation(
            db=db,
            conversation_id=chat_id,
            user_id=payload.user_id,
            sender="USER",
            message_type="AUDIO",
            message=mp3_filename.name,
            transcription=response.get("transcription")  # Save transcription
        )
        
        if not user_message:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Save AI response
        ai_message = services.add_message_to_conversation(
            db=db,
            conversation_id=chat_id,
            user_id=payload.user_id,
            sender="AI",
            message_type="TEXT",
            message=response.get("message")
        )
        
        return {
            "user_message": {
                "id": str(user_message.id),
                "audio_filename": user_message.message,
                "transcribed_message": user_message.transcription,  # Return saved transcription
                "created_at": user_message.created_at,
            },
            "ai_message": {
                "id": str(ai_message.id),
                "message": ai_message.message,
                "created_at": ai_message.created_at,
            },
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending audio message: {str(e)}")


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