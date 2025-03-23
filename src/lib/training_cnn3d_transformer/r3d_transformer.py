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
        out = self.sigmoid(avg_out + max_out)
        return x * out, out  # Retourne aussi la carte d'attention


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
        return x * attn_sigmoid, attn_sigmoid


class CBAM3D(nn.Module):
    def __init__(self, in_planes):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x, ca_attention = self.ca(x)
        x, sa_attention = self.sa(x)
        return x, sa_attention


class r3d_transformer(nn.Module):
    def __init__(self):
        super(r3d_transformer, self).__init__()
        # Chargement du backbone R3D-18 pré-entraîné
        r3d = r3d_18(pretrained=True)

        # Extraction des couches individuelles
        self.stem = r3d.stem
        self.layer1 = r3d.layer1  # Gardons les layers séparés
        self.layer2 = r3d.layer2
        self.layer3 = r3d.layer3
        self.layer4 = r3d.layer4
        self.avgpool = r3d.avgpool

        # Ajout des modules CBAM3D
        self.cbam1 = CBAM3D(64)  # 64 canaux après layer1
        self.cbam2 = CBAM3D(128)  # 128 canaux après layer2

        # Couche finale
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        attention_maps = {}

        # Passage à travers le stem
        x = self.stem(x)

        # Layer 1 + CBAM
        x = self.layer1(x)
        x, attn1 = self.cbam1(x)
        attention_maps["layer1"] = attn1

        # Layer 2 + CBAM
        x = self.layer2(x)
        x, attn2 = self.cbam2(x)
        attention_maps["layer2"] = attn2

        # Couches restantes sans attention
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification finale
        x = self.fc(x)

        return x, attention_maps
