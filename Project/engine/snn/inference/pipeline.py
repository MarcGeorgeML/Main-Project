import sys
from pathlib import Path
from typing import List, Dict, Union

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

# ── project imports ──────────────────────────────────────────────────────────
from Model.spikformer import Spikformer                          # models/
from Model.SpikEmo_Model import SpikEmo                          # models/
from preprocessing.whisper_segmenter import WhisperSegmenter             # preprocessing/
from preprocessing.video_decoder import extract_audio, extract_frames   # preprocessing/
from preprocessing.text_features import TextFeatureExtractor             # preprocessing/
from preprocessing.audio_features import AudioFeatureExtractor           # preprocessing/
from preprocessing.visual_features import VisualFeatureExtractor         # preprocessing/

from feature_utils import extract_all_features, collate_for_inference
from result_types import UtterancePrediction, EMOTION_LABELS


class InferencePipeline:
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        model_config: dict,
        weights_path: Union[str, Path],
        whisper_model_size: str = "base",
    ):
        # ── Stage 1a : GPU check ─────────────────────────────────────────────
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not found. This pipeline requires a GPU. "
                "Please run on a machine with a CUDA-capable GPU."
            )
        self.device = "cuda"
        print(f"[pipeline] GPU detected: {torch.cuda.get_device_name(0)}")

        # ── Stage 1a : enforce determinism
        # CUDA ops (convolutions, atomics) are non-deterministic by default.
        # These four lines mirror the set_seed() call in the training script
        # and guarantee identical outputs for identical inputs every run.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        # ── Stage 1b : build model architecture ─────────────────────────────
        print("[pipeline] Building model architecture …")
        spikformer = Spikformer(
            depths     = model_config["depths"],
            tau        = model_config["tau"],
            common_thr = model_config["common_thr"],
            dim        = model_config["model_dim"],
            T          = model_config["T"],
            heads      = model_config["heads"],
            qk_scale   = model_config.get("qk_scale", 0.125),
        )

        self.model = SpikEmo(
            dataset          = model_config["dataset"],
            multi_attn_flag  = model_config["multi_attn_flag"],
            roberta_dim      = model_config["roberta_dim"],
            hidden_dim       = model_config["hidden_dim"],
            dropout          = model_config["dropout"],
            num_layers       = model_config["num_layers"],
            model_dim        = model_config["model_dim"],
            num_heads        = model_config["num_heads"],
            D_m_audio        = model_config["D_m_audio"],
            D_m_visual       = model_config["D_m_visual"],
            n_classes        = model_config["n_classes"],
            spikformer_model = spikformer,
        )

        # ── Stage 1c : load weights ──────────────────────────────────────────
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        print(f"[pipeline] Loading weights from {weights_path} …")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("[pipeline] Model ready.")

        # ── Stage 1d : initialise feature extractors ─────────────────────────
        print("[pipeline] Loading feature extractors …")
        self.segmenter      = WhisperSegmenter(model_size=whisper_model_size, device=self.device)
        self.text_encoder   = TextFeatureExtractor(device=self.device)
        self.audio_encoder  = AudioFeatureExtractor(device=self.device)
        self.visual_encoder = VisualFeatureExtractor(device=self.device)
        print("[pipeline] All extractors ready.\n")

    # --------------------------------------------------------------- predict
    def predict(self, video_path: str) -> List[UtterancePrediction]:
        """
        Run end-to-end inference on a single MP4 file.

        Parameters
        ----------
        video_path : str
            Absolute or relative path to the input .mp4 file.

        Returns
        -------
        List[UtterancePrediction]
            One result per utterance detected by Whisper.
        """

        video_path = str(video_path)

        # ── Stage 2 : segment ────────────────────────────────────────────────
        utterances = self._segment(video_path)

        # ── Stage 3 : extract audio waveform once for the whole video ────────
        print("[pipeline] Decoding audio waveform …")
        audio_waveform = extract_audio(video_path)

        # ── Stage 3 : extract features ───────────────────────────────────────
        print("[pipeline] Extracting multimodal features …")
        text_embs, audio_embs, visual_embs, valid_utts = extract_all_features(
            utterances     = utterances,
            video_path     = video_path,
            audio_waveform = audio_waveform,
            text_encoder   = self.text_encoder,
            audio_encoder  = self.audio_encoder,
            visual_encoder = self.visual_encoder,
            extract_frames_fn = extract_frames,
            device         = self.device,
        )
        print(f"[pipeline] {len(valid_utts)} utterances encoded successfully.")

        # ── Stage 4 : collate ────────────────────────────────────────────────
        texts, audios, visuals, _, _ = collate_for_inference(
            text_embs, audio_embs, visual_embs, device=self.device
        )

        # ── Stage 5 : infer ──────────────────────────────────────────────────
        probs = self._infer(texts, audios, visuals)   # [B, n_classes]

        # ── Stage 6 : format ─────────────────────────────────────────────────
        return self._format_results(valid_utts, probs)

    # ---------------------------------------------------------------- helpers

    def _segment(self, video_path: str) -> List[Dict]:
        """Run Whisper and return a list of utterance dicts."""
        print("[pipeline] Segmenting audio with Whisper …")
        utterances = self.segmenter.segment_audio(video_path)
        if not utterances:
            raise RuntimeError(
                "Whisper found no speech segments in the video. "
                "Please check that the video contains audible speech."
            )
        print(f"[pipeline] {len(utterances)} utterances detected.")
        return utterances

    def _infer(
        self,
        texts: torch.Tensor,
        audios: torch.Tensor,
        visuals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through SpikEmo, return softmax probabilities [B, n_classes].
        """
        print("[pipeline] Running model inference ...")
        functional.reset_net(self.model)
        with torch.no_grad():
            _, _, _, _, mlp_outputs = self.model(texts, audios, visuals)
        # mlp_outputs : [B, n_classes]
        probs = F.softmax(mlp_outputs, dim=-1)
        return probs

    def _format_results(
        self,
        utterances: List[Dict],
        probs: torch.Tensor,
    ) -> List[UtterancePrediction]:
        """
        Convert raw probability tensors into UtterancePrediction objects.
        """
        results = []
        probs_cpu = probs.cpu()

        for utt, prob_vec in zip(utterances, probs_cpu):
            class_index = int(prob_vec.argmax().item())
            confidence  = float(prob_vec[class_index].item())
            all_scores  = {
                EMOTION_LABELS[i]: round(float(prob_vec[i].item()), 4)
                for i in range(len(EMOTION_LABELS))
            }
            results.append(
                UtterancePrediction(
                    start       = utt["start"],
                    end         = utt["end"],
                    text        = utt["text"],
                    emotion     = EMOTION_LABELS[class_index],
                    class_index = class_index,
                    confidence  = confidence,
                    all_scores  = all_scores,
                )
            )

        print(f"[pipeline] Done — {len(results)} predictions returned.\n")
        return results