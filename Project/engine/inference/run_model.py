# engine/inference/run_model.py
import torch
from Model.SpikEmo_Model import SpikEmo  # adjust import to your actual model class

def load_snn_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    args = checkpoint["args"]

    # -----------------------------
    # EXPLICIT constructor mapping
    # -----------------------------
    model = SpikEmo(
        num_layers=args["num_layers"],
        model_dim=args["model_dim"],
        num_heads=args["num_heads"],
        hidden_dim=args["hidden_dim"],
        dataset=args["dataset"]  # include ONLY if your model expects it
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def run_snn_inference(model, text, audio, video):
    logits = model(text, audio, video)
    return torch.argmax(logits, dim=-1).item()
