import torch
from transformers import RobertaTokenizer, RobertaModel


class TextFeatureExtractor:

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.model.to(self.device)  # type: ignore[reportArgumentType]
        self.model.eval()

    def encode_batch(self, texts):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden * mask
            summed = masked_hidden.sum(1)
            counts = mask.sum(1)
            embeddings = summed / counts
        return embeddings

    def encode(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            emb = self.encode_batch(batch)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)
