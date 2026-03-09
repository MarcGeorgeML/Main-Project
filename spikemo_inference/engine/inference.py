import torch
from spikingjelly.activation_based import functional


class SpikeMoInference:

    def __init__(self, model, checkpoint_path, device):
        self.device = device
        self.model = model.to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.label_map = [
            "happiness",
            "sadness",
            "neutral",
            "anger",
            "excitement",
            "frustration",
        ]

    @torch.no_grad()
    def predict(self, inputs, return_labels=False):

        inputs = [x.to(self.device) for x in inputs]
        text_out, audio_out, visual_out, fc, logits = self.model(*inputs)
        logits[:, 2] -= 1.2
        logits[:, 5] -= 1.0
        logits[:,0] += 0.4   # happiness
        logits[:,4] += 0.3   # excitement
        print("Text feature norm:", text_out.norm().item())
        print("Audio feature norm:", audio_out.norm().item())
        print("Visual feature norm:", visual_out.norm().item())
        print("Logits: ", logits)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        confidences = probs[torch.arange(len(preds), device=self.device), preds]
        functional.reset_net(self.model)

        preds_list = preds.tolist()
        conf_list = confidences.tolist()
        smoothed = []

        for i in range(len(preds_list)):

            window_labels = []
            window_conf = []

            for j in range(max(0, i - 1), min(len(preds_list), i + 2)):
                window_labels.append(preds_list[j])
                window_conf.append(conf_list[j])

            counts = {label: window_labels.count(label) for label in set(window_labels)}
            majority_label = max(counts, key=counts.get)

            if list(counts.values()).count(counts[majority_label]) > 1:
                max_conf_idx = window_conf.index(max(window_conf))
                majority_label = window_labels[max_conf_idx]

            smoothed.append(majority_label)

        if return_labels:

            results = []

            for p, c in zip(smoothed, conf_list):
                label = self.label_map[p]
                results.append((label, float(c)))

            return results

        return preds.new_tensor(smoothed)
