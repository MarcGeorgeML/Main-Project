import torch

@torch.no_grad()
def extract_text_features(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    cls = outputs.last_hidden_state[:, 0, :]  # (1, 768)

    return cls.unsqueeze(0)  # (T=1, B=1, F)
