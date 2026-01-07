import whisper
import tempfile
from data.minio_client import minio_client, MINIO_BUCKET

_model = whisper.load_model("base")

def transcribe_audio(audio_object_key: str) -> str:
    """
    Download audio from MinIO and transcribe using Whisper.
    """
    response = None
    try:
        response = minio_client.get_object(
            MINIO_BUCKET,
            audio_object_key
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            tmp.write(response.read())
            tmp.flush()

            result = _model.transcribe(tmp.name)
            return result.get("text", "").strip()

    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""

    finally:
        if response:
            response.close()
            response.release_conn()
