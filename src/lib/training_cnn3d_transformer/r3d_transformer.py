import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.nn import Transformer


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_max_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(avg_max_out)
        attn_sigmoid = self.sigmoid(attn)
        return x * attn_sigmoid


class CBAM3D(nn.Module):
    def __init__(self, in_planes):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        attention_map = torch.mean(x, dim=1, keepdim=True)
        return x, attention_map


class r3d_transformer(nn.Module):
    def __init__(self):
        super(r3d_transformer, self).__init__()
        self.backbone = r3d_18(pretrained=True)

        # Ajout du CBAM3D aux couches ResNet3D
        self.cbam1 = CBAM3D(64)
        self.cbam2 = CBAM3D(128)

        self.backbone.layer1 = nn.Sequential(self.backbone.layer1, self.cbam1)
        self.backbone.layer2 = nn.Sequential(self.backbone.layer2, self.cbam2)

        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        attention_maps = {}

        # Passage dans la couche 1 + CBAM3D
        x = self.backbone.stem(x)
        x = self.backbone.layer1[0](x)  # RésNet3D layer1
        x, attn1 = self.cbam1(x)  # CBAM3D de layer1
        attention_maps["layer1"] = attn1

        # Passage dans la couche 2 + CBAM3D
        x = self.backbone.layer2[0](x)  # RésNet3D layer2
        x, attn2 = self.cbam2(x)  # CBAM3D de layer2
        attention_maps["layer2"] = attn2

        # Passage dans les couches finales de ResNet3D
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Prédiction finale
        x = self.fc(x)

        return x, attention_maps
