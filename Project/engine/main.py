from data.redis_client import RedisClient
from schemas.requests import RequestType
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── Add snn/inference to path so pipeline imports work ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "snn", "inference"))

from snn.inference.pipeline import InferencePipeline

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

    # ── Load model once at startup ───────────────────────────────────────────
    print("[main] Loading inference pipeline...")
    model_config = load_config(CONFIG_PATH)
    pipeline = InferencePipeline(
        model_config=model_config,
        weights_path=WEIGHTS_PATH,
        whisper_model_size="base",
    )
    print("[main] Pipeline ready.\n")

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

                        results = pipeline.predict(video_path)

                        # Use the highest-confidence utterance as the overall prediction
                        best = max(results, key=lambda r: r.confidence)

                        response = {
                            "request_id": request.request_id,
                            "emotion":     best.emotion,
                            "confidence":  round(best.confidence, 4),
                            "transcription": best.text,
                            "all_scores":  best.all_scores,
                            "message": "Thanks for sharing. I'm here to listen."
                        }

                    else:
                        response = {
                            "request_id": request.request_id,
                            "error": "invalid request type"
                        }

                    print("Request ID :", request.request_id)
                    print("User       :", data.user_id)
                    print("Emotion    :", response.get("emotion", "N/A"))
                    print("Confidence :", response.get("confidence", "N/A"))
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
