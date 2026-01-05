# engine/loop/redis_loop.py
import json
from data.redis_client import RedisClient
from models.requests import RequestType
from utils.whisper_transcriber import transcribe_audio

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features
from inference.validate_inputs import validate_modalities
from inference.run_model import run_snn_inference

ACK_CHANNEL = "ack_channel"
JOB_STREAM = "job"

LABEL_MAP = {
    0: "happiness",
    1: "sadness",
    2: "neutral",
    3: "anger",
    4: "excitement",
    5: "frustration"
}

def start_redis_loop(
    snn_model,
    tokenizer,
    text_model,
    smile,
    groq_fn,
    device
):
    client = RedisClient()
    client.clear_stream(JOB_STREAM)

    while True:
        messages = client.read_stream(JOB_STREAM)

        for message_id, message_data in messages:
            if isinstance(message_data.get("data"), str):
                message_data["data"] = json.loads(message_data["data"])

            request = RequestType(**message_data)
            data = request.data

            # -----------------------------
            # Text source
            # -----------------------------
            if data.type == "audio":
                text = transcribe_audio(data.data)
            else:
                text = data.data

            # -----------------------------
            # Feature extraction
            # -----------------------------
            text_feat = extract_text_features(
                text, tokenizer, text_model, device
            )
            audio_feat = extract_audio_features(
                data.data, smile, device
            )
            video_feat = extract_video_features(device)

            validate_modalities(text_feat, audio_feat, video_feat)

            # -----------------------------
            # SNN inference
            # -----------------------------
            label_idx = run_snn_inference(
                snn_model,
                text_feat,
                audio_feat,
                video_feat
            )

            emotion_text = LABEL_MAP[label_idx]

            # -----------------------------
            # Groq call (UNCHANGED PROMPT)
            # -----------------------------
            groq_response = groq_fn(
                f"The detected emotion is {emotion_text}. "
                f"User message: {text}"
            )

            client.publish_ack(
                ACK_CHANNEL,
                {
                    "request_id": request.request_id,
                    "emotion": emotion_text,
                    "message": groq_response
                }
            )
