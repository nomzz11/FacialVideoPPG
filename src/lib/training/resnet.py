import torch.nn as nn
import torchvision.models as models


class ResNetPPG(nn.Module):
    def __init__(self):
        super(ResNetPPG, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)
