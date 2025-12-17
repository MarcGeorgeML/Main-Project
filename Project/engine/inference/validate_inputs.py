def validate_modalities(text, audio, video):
    assert text.ndim == audio.ndim == video.ndim == 3
    assert text.size(0) == audio.size(0) == video.size(0)
