from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(r"c:\Users\Marc\Desktop\Programming\Research\spikemo_inference\engine\.env"))

import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
from facenet_pytorch import MTCNN

class VisualFeatureExtractor:

    def __init__(self, device, max_frames_per_utterance=16):

        self.device = device
        self.max_frames = max_frames_per_utterance
        self.mtcnn = MTCNN(keep_all=False, device=device)
        self.proj = torch.nn.Linear(2048, 1000).to(device)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.model.eval()
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def load_video_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        return frames, fps

    def sample_frames(self, frames):

        n = len(frames)

        if n <= self.max_frames:
            return frames

        indices = np.linspace(0, n - 1, self.max_frames).astype(int)

        return [frames[i] for i in indices]

    def detect_face_bbox(self, frames):

        search_frames = frames[:3]

        for frame in search_frames:

            result = self.mtcnn.detect(frame)
            boxes = result[0] if isinstance(result, tuple) else result

            if boxes is not None:
                return boxes[0].astype(int)

        return None

    def crop_with_bbox(self, frame, bbox):

        if bbox is None:
            return frame

        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        return frame[y1:y2, x1:x2]

    def extract_batch(self, frames_list):

        outputs = []
        
        for frames in frames_list:

            if len(frames) == 0:
                outputs.append(torch.zeros(1000).to(self.device))
                continue

            frames = self.sample_frames(frames)
            bbox = self.detect_face_bbox(frames)
            imgs = []

            for f in frames:
                face = self.crop_with_bbox(f, bbox)
                imgs.append(self.transform(face))
            imgs = torch.stack(imgs).to(self.device)

            with torch.no_grad():
                out = self.model(imgs)
            out = out.view(out.size(0), -1)
            out = out.mean(dim=0)
            out = self.proj(out)
            outputs.append(out)

        return torch.stack(outputs)
