from data.redis_client import RedisClient
from schemas.requests import RequestType
import json
import os
from groq import Groq
from dotenv import load_dotenv
from snn.emotion_infer import infer_emotion
from utils.whisper_transcriber import transcribe_audio 
from utils.resolve_audio_path import resolve_audio_path

# Load environment variables from .env file
load_dotenv()

ACK_CHANNEL = "ack_channel"
JOB_STREAM = "job"

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_sentiment(text):
    """
    Analyze the emotional sentiment of the text using Groq API.
    Returns a short response message.
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an emotional sentiment analysis chatbot. "
                        "Analyze the user's message and respond with a short, "
                        "empathetic message (1-2 sentences) that acknowledges "
                        "their emotional state and provides supportive feedback."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=100
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "I'm having trouble analyzing that right now. Please try again."


def main():
    print("\n=================================")
    print("Starting Senticore Engine...")
    print("=================================\n")

    client = RedisClient()
    client.clear_stream(JOB_STREAM)

    try:
        while True:
            messages = client.read_stream(JOB_STREAM)

            for message_id, message_data in messages:
                try:
                    if isinstance(message_data.get("data"), str):
                        message_data["data"] = json.loads(message_data["data"])

                    request = RequestType(**message_data)
                    data = request.data

                    # if data.type == "audio":
                    #     transcribed_text = transcribe_audio(data.data)

                    #     audio_path = resolve_audio_path(data.data)

                    #     emotion_result = infer_emotion(audio_path)

                    #     emotion = emotion_result["prediction"]
                    #     probs = emotion_result["probabilities"]

                    #     ai_response = analyze_sentiment(transcribed_text)

                    #     response = {
                    #         "request_id": request.request_id,
                    #         "type": "audio",
                    #         "transcription": transcribed_text,
                    #         "emotion": emotion,
                    #         "probabilities": probs,
                    #         "message": f"{ai_response}"
                    #     }



                    # elif data.type == "text":
                    #     ai_response = analyze_sentiment(data.data)

                    #     response = {
                    #         "request_id": request.request_id,
                    #         "type": "text",
                    #         "message": ai_response
                    #     }

                    if data.type == "video":
                        print("-" * 60)
                        print("got video")
                        print("-" * 60)

                        response = {
                            "request_id": request.request_id,
                            "transcription": "This is a dummy test.",
                            "emotion": "neutral",
                            "confidence": 0.75,
                            "message": "Thanks for sharing. I'm here to listen."
                        }
                    else:
                        response = {
                            "request_id": request.request_id,
                            "error": "invalid request"
                        }

                    print("Request Received:", message_id)
                    print("User:", data.user_id)
                    print("Message:", data.data)
                    print("AI Response:", response.get("message", "N/A"))
                    print("-" * 40)

                    client.publish_ack(ACK_CHANNEL, response)

                except Exception as e:
                    print(f"Error processing message: {e}")
                    print("Raw message:", message_data)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        client.close()
        print("Service stopped.")


if __name__ == "__main__":
    main()
