# import whisper
# import os
# from .resolve_audio_path import resolve_audio_path

# _model = whisper.load_model("base")

# ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(ENGINE_DIR, "..", ".."))
# AUDIO_BASE_DIR = os.path.join(PROJECT_ROOT, "temp")



# def transcribe_audio(audio_ref: str) -> str:
#     """
#     Transcribe an audio file using local Whisper.
#     """
#     try:
#         audio_path = resolve_audio_path(audio_ref)
#         result = _model.transcribe(audio_path)
#         return result.get("text", "").strip()

#     except Exception as e:
#         print(f"Whisper transcription error: {e}")
#         return ""
