import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.nn import Transformer


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        # Convolution sur la concaténation des deux cartes attention (max et moyenne)
        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calcul du max et du moyen sur les canaux (C) de la dimension spatiale
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Moyenne sur les canaux
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max sur les canaux

        # Concaténation des deux sorties (moyenne et max) pour capturer plus d'informations
        out = torch.cat([avg_out, max_out], dim=1)

        # Application de la convolution et de la fonction sigmoïde
        out = self.conv1(out)
        out = self.sigmoid(out)

        # Retourner l'entrée modifiée avec l'attention spatiale
        return x * out


class r3d_transformer(nn.Module):
    def __init__(self):
        super(r3d_transformer, self).__init__()
        # Load pre-trained ResNet3D-18 model on videos to extract features
        self.backbone = r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Identity()

        # Ajouter l'attention spatiale
        self.spatial_attention = SpatialAttention(kernel_size=3)

        # FC layer to predict final PPG value
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.backbone(x)
        print(x.shape)
        x = self.spatial_attention(x)
        x = self.fc(x)
        print(x.shape)
        x = x.squeeze(-1)
        return x
