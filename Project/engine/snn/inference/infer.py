"""
python inference/infer.py
"""

import argparse
from pipeline import InferencePipeline
from pathlib import Path
import json

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.pop("_meta", None)
    print(f"[infer] Config loaded from {path}")
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpikEmo — Multimodal Emotion Inference"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to the input .mp4 file",
        default="data_sorted/disgust/video/disgust_00008.mp4"
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to the saved model weights (.pt checkpoint file)",
        default="checkpoints/SpikEmo_T8_dim256_h4_lr4.9334886565290195e-05_bs32_20260323_124137/checkpoint_epoch37_f10.7924.pt"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to inference_config.json saved during training",
        default="configs/inference_config.json"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size to use for segmentation (default: base)",
    )
    return parser.parse_args()


def print_results(results):
    print("\n" + "=" * 60)
    print(f"  RESULTS  —  {len(results)} utterance(s) detected")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r.start:.2f}s → {r.end:.2f}s")
        print(f"       Text       : \"{r.text}\"")
        print(f"       Prediction : {r.emotion.upper()}  ({r.confidence * 100:.1f}%)")
        print("       All scores :")
        for emotion, score in sorted(r.all_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 30)
            print(f"         {emotion:<10} {score * 100:5.1f}%  {bar}")

    print("\n" + "=" * 60 + "\n")


def main():
    args = parse_args()

    # ── load config from JSON saved during training ──────────────────────────
    model_config = load_config(args.config)

    # ── initialise pipeline ──────────────────────────────────────────────────
    pipeline = InferencePipeline(
        model_config       = model_config,
        weights_path       = args.weights,
        whisper_model_size = args.whisper_model,
    )

    # ── run inference ────────────────────────────────────────────────────────
    results = pipeline.predict(args.video)

    # ── display results ──────────────────────────────────────────────────────
    print_results(results)


if __name__ == "__main__":
    main()