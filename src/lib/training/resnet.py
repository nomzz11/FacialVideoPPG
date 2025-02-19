import torch.nn as nn
import torchvision.models as models


class ResNetPPG(nn.Module):
    def __init__(self):
        super(ResNetPPG, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        batch_size, time_steps, c, h, w = x.size()
        x_resnet = x.view(batch_size * time_steps, c, h, w)
        x_resnet = self.resnet(x_resnet)
        x_resnet = x_resnet.view(batch_size, time_steps, 1)

        return x_resnet
