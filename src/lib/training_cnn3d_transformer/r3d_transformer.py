import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.nn import Transformer


class r3d_transformer(nn.Module):
    def __init__(self):
        super(r3d_transformer, self).__init__()
        # Load pre-trained ResNet3D-18 model on videos to extract features
        self.backbone = r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Identity()

        # Transformer to capture temporal relationships
        self.transformer = Transformer(d_model=512, nhead=8, num_encoder_layers=6)

        # FC layer to predict final PPG value
        self.fc = nn.Linear(512, 90)

    def forward(self, x):
        x = self.backbone(x)
        x = self.feature_adapter(x)
        src, tgt = x, x
        x = self.transformer(src, tgt)
        x = self.fc(x)
        return x
