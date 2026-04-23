import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class FeatureExtractor(nn.Module):
    """
    ResNet-50 backbone projecting detection crops to 128-d L2-normalised embeddings.
    Weights are ImageNet-pretrained; no re-ID fine-tuning required for basic use.
    """

    def __init__(self, model_path=None, device="cpu"):
        super().__init__()
        self.device = device

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop classifier
        self.projection = nn.Linear(2048, 128)

        if model_path:
            state = torch.load(model_path, map_location=device)
            self.load_state_dict(state)

        self.to(device)
        self.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, frame, detections):
        """Return list of 128-d feature arrays aligned with detections list."""
        if not detections:
            return []

        patches = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.to_tlbr())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                patches.append(torch.zeros(3, 128, 64))
                continue
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            patches.append(self.transform(crop))

        batch = torch.stack(patches).to(self.device)
        feats = self.backbone(batch).squeeze(-1).squeeze(-1)
        feats = self.projection(feats)
        feats = nn.functional.normalize(feats, dim=1)
        return [feats[i].cpu().numpy() for i in range(feats.shape[0])]
