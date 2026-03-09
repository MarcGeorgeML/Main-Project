from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(r"c:\Users\Marc\Desktop\Programming\Research\spikemo_inference\engine\.env"))

import torch
from transformers import RobertaTokenizer, RobertaModel

class TextFeatureExtractor:

    def __init__(self, device):
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base").to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_batch(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)
        return features
