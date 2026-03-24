import av
import numpy as np
import subprocess
from typing import cast
from av.container.input import InputContainer


def extract_audio(video_path, sr=16000):

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "-",
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True
    )

    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def extract_frames(video_path, start_time, end_time, max_frames=16):

    if end_time <= start_time:
        return []

    with cast(InputContainer, av.open(video_path, mode="r")) as container:
        stream = container.streams.video[0]

        if stream.time_base is not None:
            seek_pts = int(start_time / float(stream.time_base))
            container.seek(
                max(0, seek_pts), stream=stream, backward=True, any_frame=False
            )

        frames = []

        for frame in container.decode(video=stream.index):
            pts = frame.pts
            tb = stream.time_base
            if pts is None or tb is None:
                continue

            timestamp = float(pts * tb)

            if timestamp < start_time:
                continue
            if timestamp > end_time:
                break

            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        return []

    indices = np.linspace(
        0,
        len(frames) - 1,
        num=min(max_frames, len(frames)),
        dtype=int,
    )
    return [frames[i] for i in indices]
