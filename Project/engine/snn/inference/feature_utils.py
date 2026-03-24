from typing import List, Dict, Tuple
import torch


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(
    utterances: List[Dict],
    video_path: str,
    audio_waveform,           # np.ndarray float32 at 16 kHz
    text_encoder,             # TextFeatureExtractor
    audio_encoder,            # AudioFeatureExtractor
    visual_encoder,           # VisualFeatureExtractor
    extract_frames_fn,        # video_decoder.extract_frames
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    valid_texts   = []
    valid_audios  = []
    valid_frames  = []
    valid_utts    = []

    for utt in utterances:
        start = utt["start"]
        end   = utt["end"]
        text  = utt["text"].strip()

        if not text:
            continue

        try:
            # --- audio slice --------------------------------------------------
            audio_segment = audio_encoder.slice_segment(audio_waveform, start, end)
            if len(audio_segment) == 0:
                print(f"  [warn] empty audio slice  {start:.2f}s → {end:.2f}s, skipping")
                continue

            # --- video frames -------------------------------------------------
            frames = extract_frames_fn(video_path, start, end)
            if len(frames) == 0:
                print(f"  [warn] no frames decoded  {start:.2f}s → {end:.2f}s, skipping")
                continue

            valid_texts.append(text)
            valid_audios.append(audio_segment)
            valid_frames.append(frames)
            valid_utts.append(utt)

        except Exception as exc:
            print(f"  [warn] feature extraction failed for segment "
                  f"{start:.2f}s → {end:.2f}s : {exc}")

    if len(valid_texts) == 0:
        raise RuntimeError(
            "No valid utterances could be encoded. "
            "Check that the video has audible speech and decodable frames."
        )

    # --- encode text (batch) --------------------------------------------------
    text_embs = text_encoder.encode(valid_texts)           # [B, 768]

    # --- encode audio (batch) -------------------------------------------------
    audio_embs = audio_encoder.encode(valid_audios)        # [B, 768]

    # --- encode visual (one segment at a time, then stack) --------------------
    visual_list = []
    for frames in valid_frames:
        emb = visual_encoder.encode(frames)                # [2048]
        visual_list.append(emb)
    visual_embs = torch.stack(visual_list, dim=0)          # [B, 2048]

    return text_embs, audio_embs, visual_embs, valid_utts


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_for_inference(
    text_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    visual_embs: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    B = text_embs.shape[0]

    # add sequence/time dimension so shape is [1, B, D]
    texts   = text_embs.unsqueeze(0).to(device)
    audios  = audio_embs.unsqueeze(0).to(device)
    visuals = visual_embs.unsqueeze(0).to(device)

    # masks — all ones (single video, no padding)
    speaker_mask   = torch.ones(1, B, 1,  device=device)
    utterance_mask = torch.ones(B, 1,     device=device)

    return texts, audios, visuals, speaker_mask, utterance_mask