import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class AudioFeatureExtractor:

    def __init__(self, device="cuda", batch_size=16):

        self.device = device
        self.batch_size = batch_size
        self.sample_rate = 16000

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        self.model.to(self.device)  # type: ignore[reportArgumentType]
        self.model.eval()

    def slice_segment(self, audio, start, end):
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)
        return audio[start_sample:end_sample]

    def encode_batch(self, waveforms):
        kwargs = {
            "sampling_rate": self.sample_rate,
            "return_tensors": "pt",
            "padding": True,
            "return_attention_mask": True,
        }
        inputs = self.processor(waveforms, **kwargs)
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)
        return pooled

    def encode(self, waveforms):

        results = []

        for i in range(0, len(waveforms), self.batch_size):
            batch = waveforms[i : i + self.batch_size]
            emb = self.encode_batch(batch)
            results.append(emb.cpu())

        return torch.cat(results, dim=0)






