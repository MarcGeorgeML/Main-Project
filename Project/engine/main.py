from data.redis_client import RedisClient
from schemas.requests import RequestType
import json
import os
import sys
from dotenv import load_dotenv
# from schemas.requests import ChatHistoryItem as HistoryItem


load_dotenv()

# ── Add snn/inference to path so pipeline imports work ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "snn", "inference"))

from snn.inference.pipeline import InferencePipeline
from llm import LLMChatPipeline

ACK_CHANNEL = "ack_channel"
JOB_STREAM = "job"

CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "snn", "config", "inference_config.json")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "snn", "inference", "senticore-model.pt")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.pop("_meta", None)
    return cfg


def main():
    print("\n=================================")
    print("Starting Senticore Engine...")
    print("=================================\n")

    # ── Load SNN inference pipeline ─────────────────────────────────────────
    print("[main] Loading inference pipeline...")
    model_config = load_config(CONFIG_PATH)
    snn_pipeline = InferencePipeline(
        model_config=model_config,
        weights_path=WEIGHTS_PATH,
        whisper_model_size="base",
    )
    print("[main] SNN pipeline ready.")

    # ── Load LLM chat pipeline ───────────────────────────────────────────────
    print("[main] Loading LLM chat pipeline...")
    llm_pipeline = LLMChatPipeline(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )
    print("[main] LLM pipeline ready.\n")

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

                    if data.type == "video":
                        print("-" * 60)
                        video_path = data.data
                        print(f"[main] Processing video: {video_path}")

                        # ── 1. SNN emotion inference ─────────────────────────
                        results = snn_pipeline.predict(video_path)
                        best = max(results, key=lambda r: r.confidence)

                        current_emotion    = best.emotion
                        current_confidence = round(best.confidence, 4)
                        transcription      = best.text

                        # ── 2. Print all emotions and their confidence scores ─
                        print("\n[Emotion Scores]")
                        if best.all_scores:
                            # Sort by score descending for readability
                            sorted_scores = sorted(
                                best.all_scores.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            for emotion_label, score in sorted_scores:
                                bar = "█" * int(score * 20)  # simple visual bar (max 20 chars)
                                print(f"  {emotion_label:<10} {score:.4f}  {bar}")
                        print(f"  → Best: {current_emotion} ({current_confidence:.4f})")

                        # ── 3. Build history list for LLM ────────────────────
                        # history comes from the backend as a list of dicts:
                        # [{ user_message, ai_response, emotion, confidence }, ...]
                        history = data.history if hasattr(data, "history") and data.history else []

                        # ── 4. Generate empathetic LLM response ──────────────
                        ai_message = llm_pipeline.generate_response(
                            transcription=transcription,
                            current_emotion=current_emotion,
                            current_confidence=current_confidence,
                            history=history,
                        )

                        response = {
                            "request_id":   request.request_id,
                            "emotion":       current_emotion,
                            "confidence":    current_confidence,
                            "transcription": transcription,
                            "all_scores":    best.all_scores,
                            "message":       ai_message,
                        }

                    else:
                        response = {
                            "request_id": request.request_id,
                            "error": "invalid request type"
                        }

                    print("\nRequest ID   :", request.request_id)
                    print("User         :", data.user_id)
                    print("Transcription:", response.get("transcription", "N/A"))
                    print("Emotion      :", response.get("emotion", "N/A"))
                    print("Confidence   :", response.get("confidence", "N/A"))
                    print("-" * 60)

                    client.publish_ack(ACK_CHANNEL, response)

                except Exception as e:
                    print(f"[main] Error processing message {message_id}: {e}")
                    client.publish_ack(ACK_CHANNEL, {
                        "request_id": message_data.get("request_id", "unknown"),
                        "error": str(e)
                    })

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        client.close()
        print("Service stopped.")


if __name__ == "__main__":
    main()