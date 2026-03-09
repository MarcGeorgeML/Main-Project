import subprocess
import numpy as np


def load_audio_from_video(video_path, sr=16000):

    command = [
        "ffmpeg",
        "-threads",
        "0",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "-loglevel",
        "quiet",
        "-",
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    audio_bytes, _ = process.communicate()
    audio = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

    return audio, sr
