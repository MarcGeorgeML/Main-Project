import torch

def build_tensors(texts, audios, visuals, device):
    
    if isinstance(texts, torch.Tensor):
        texts = list(texts)

    if isinstance(audios, torch.Tensor):
            audios = list(audios)

    if isinstance(visuals, torch.Tensor):
            visuals = list(visuals)

    T = len(texts)
    text_tensor = torch.stack(texts).unsqueeze(1).to(device)
    audio_tensor = torch.stack(audios).unsqueeze(1).to(device)
    visual_tensor = torch.stack(visuals).unsqueeze(1).to(device)

    speaker_mask = (
        torch.tensor([[1, 0]] * T, dtype=torch.float32).unsqueeze(1).to(device)
    )

    utterance_mask = torch.ones(1, T).to(device)
    padded_labels = torch.zeros(T).long().to(device)

    return (
        text_tensor,
        audio_tensor,
        visual_tensor,
        speaker_mask,
        utterance_mask,
        padded_labels,
    )
