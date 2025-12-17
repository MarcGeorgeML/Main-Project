import torch
import numpy as np
import opensmile
import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import os

class SpikEmoInferenceDataset(Dataset):
    def __init__(self, audio_path, transcript, audio_dim=1582, visual_dim=35, tokenizer=None, text_model=None):
        """
        Extract IEMOCAP-matching features for SpikEmo inference
        
        Args:
            audio_path: path to .mp3/.wav
            transcript: text string
            audio_dim: target dim from your AudioFeatures.pkl (usually 1582)
            visual_dim: FACET dim (usually 35)
        """
        self.audio_path = audio_path
        self.transcript = transcript
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        
        # Load extractors (shared across batches)
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2013,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        self.tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.text_model = AutoModel.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.text_model.eval()
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # 1. EXTRACT AUDIO FEATURES (OpenSMILE ComParE → 1582)
        audio_feat = self._extract_audio_features()
        
        # 2. EXTRACT TEXT FEATURES (EmoBERTa → 768)
        text_feat = self._extract_text_features()
        
        # 3. DUMMY VISUAL FEATURES (zeros)
        visual_feat = np.zeros(self.visual_dim, dtype=np.float32)
        
        # 4. FORMAT AS SINGLE-UTTERANCE "VIDEO" [1, feat_dim]
        text_tensor = torch.FloatTensor(text_feat[np.newaxis, :])
        audio_tensor = torch.FloatTensor(audio_feat[np.newaxis, :])
        visual_tensor = torch.FloatTensor(visual_feat[np.newaxis, :])
        speaker_tensor = torch.FloatTensor([[0, 1]])  # Dummy female
        length_tensor = torch.FloatTensor([1.0])      # Single utterance
        label_tensor = torch.LongTensor([0])          # Dummy (ignored)
        
        return (text_tensor, audio_tensor, visual_tensor, 
                speaker_tensor, length_tensor, label_tensor)
    
    def _extract_audio_features(self):
        """OpenSMILE ComParE_2013 functionals → [audio_dim]"""
        # Load audio
        wav, sr = librosa.load(self.audio_path, sr=16000, mono=True)
        
        # Extract OpenSMILE features (handles file internally)
        feats_df = self.smile.process_signal(wav, sr)
        features = feats_df.values.flatten().astype(np.float32)
        
        # Truncate/pad to exact training dimension
        if len(features) > self.audio_dim:
            features = features[:self.audio_dim]
        else:
            features = np.pad(features, (0, self.audio_dim - len(features)), 'constant')
            
        return features
    
    def _extract_text_features(self):
        """EmoBERTa mean-pooled → [768]"""
        inputs = self.tokenizer(
            self.transcript, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return pooled.astype(np.float32)

# Your existing IEMOCAPDataset for collate_fn
class IEMOCAPDataset(Dataset):
    # ... your existing code ...
    pass
