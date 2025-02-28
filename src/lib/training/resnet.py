import torch.nn as nn
import torchvision.models as models


class ResNetPPG(nn.Module):
    def __init__(self):
        super(ResNetPPG, self).__init__()
        self.resnet = models.resnet152(pretrained=True)

        """
        for param in self.resnet.parameters():
            param.requires_grad = False
        """

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet(x)
