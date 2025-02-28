import torch
import torch.nn as nn
import torchvision.models as models


class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, sequence_length=30):
        super(ResNetLSTM, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.feature_size = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)

        features = self.resnet(x)  # (batch_size * seq_len, feature_size)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)

        output = self.fc(lstm_out)

        return output.squeeze(-1)
