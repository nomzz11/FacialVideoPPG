import torch, torch.nn as nn


class CCCLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        target_mean = torch.mean(target, dim=-1, keepdim=True)

        pred_var = torch.var(pred, dim=-1)
        target_var = torch.var(target, dim=-1)

        covariance = torch.mean((pred - pred_mean) * (target - target_mean), dim=-1)

        ccc = (
            2
            * covariance
            / (pred_var + target_var + (pred_mean - target_mean) ** 2 + self.eps)
        )
        return 1 - ccc.mean()
