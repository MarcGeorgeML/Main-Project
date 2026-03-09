import torch

from engine.segmenter import UtteranceSegmenter
from engine.text_features import TextFeatureExtractor
from engine.audio_features import AudioFeatureExtractor
from engine.visual_features import VisualFeatureExtractor
from engine.tensor_builder import build_tensors
from engine.inference import SpikeMoInference

from Model.SpikEmo_Model import SpikEmo
from Model.spikformer import Spikformer


def build_model(device):

    spikformer_model = Spikformer(
        depths=2,
        T=32,
        tau=10.0,
        common_thr=1.0,
        dim=256,
        heads=8
    )

    model = SpikEmo(
        dataset="IEMOCAP",
        multi_attn_flag=True,
        roberta_dim=768,
        hidden_dim=1024,
        dropout=0,
        num_layers=6,
        model_dim=256,
        num_heads=4,
        D_m_audio=512,
        D_m_visual=1000,
        D_g=256,
        D_p=256,
        D_e=256,
        D_h=256,
        n_classes=6,
        n_speakers=2,
        listener_state=False,
        context_attention="simple",
        D_a=256,
        dropout_rec=0,
        device=device,
        spikformer_model=spikformer_model
    )

    return model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path = "input.mp4"

    # Initialize pipeline modules
    segmenter = UtteranceSegmenter()

    text_ext = TextFeatureExtractor(device)
    audio_ext = AudioFeatureExtractor(device)
    visual_ext = VisualFeatureExtractor(device)

    model = build_model(device)

    runner = SpikeMoInference(
        model,
        "checkpoints/spikemo_best_IEMOCAP.pt",
        device
    )

    # Step 1 — Speech segmentation
    utterances = segmenter.segment(video_path)

    if len(utterances) == 0:
        print("No utterances detected.")
        return

    if len(utterances) < 2:
        print("Warning: DialogueRNN works best with multiple utterances")

    # Step 2 — Decode video frames once
    frames, fps = visual_ext.load_video_frames(video_path)

    texts_raw = []
    audio_segments = []
    frame_groups = []

    for u in utterances:

        texts_raw.append(u["text"])

        # audio slice
        audio_seg = audio_ext.load_audio_segment(
            video_path,
            u["start"],
            u["end"]
        )

        audio_segments.append(audio_seg)

        # frame slice
        start_f = int(u["start"] * fps)
        end_f = int(u["end"] * fps)

        frame_groups.append(frames[start_f:end_f])

    # Step 3 — Feature extraction
    text_feats = text_ext.extract_batch(texts_raw)

    audio_feats = audio_ext.extract_batch(audio_segments)

    visual_feats = visual_ext.extract_batch(frame_groups)

    # Step 4 — Tensor construction
    inputs = build_tensors(
        text_feats,
        audio_feats,
        visual_feats,
        device
    )

    # Step 5 — Model inference
    preds = runner.predict(inputs, return_labels=True)

    print("\nEmotion Predictions\n")

    for i, (utt, emotion) in enumerate(zip(utterances, preds)):
        label, conf = emotion
        print(f"Utterance {i+1}: {label} ({conf:.2f})")
        print(f"Text: {utt['text']}\n")


if __name__ == "__main__":
    main()