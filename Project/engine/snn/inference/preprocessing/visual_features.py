import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


class VisualFeatureExtractor:

    def __init__(self, device="cuda", batch_size=32):

        self.device = device
        self.batch_size = batch_size

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # remove classifier
        self.cnn.to(self.device)
        self.cnn.eval()

        self.feature_dim = 2048

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_frames(self, frames):
        tensors = [self.transform(frame) for frame in frames]
        return torch.stack(tensors)

    def encode_frames(self, frames):
        frames = self.preprocess_frames(frames)
        frames = frames.to(self.device)

        with torch.no_grad():
            features = self.cnn(frames)
            features = features.flatten(1)
        return features

    def encode(self, frames):

        if len(frames) == 0:
            return torch.zeros(2048)

        frame_features = self.encode_frames(frames)
        mean_feat = frame_features.mean(dim=0)
        max_feat = frame_features.max(dim=0).values
        pooled = 0.5 * mean_feat + 0.5 * max_feat
        return pooled.cpu()
