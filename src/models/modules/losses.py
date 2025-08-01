import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCEContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        B = h1.size(0)
        z = torch.cat([h1, h2], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.T) / self.temperature

        labels = torch.arange(2 * B, device=z.device)
        labels = (labels + B) % (2 * B)

        # Mask out self-similarities
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))  # use -inf to avoid exp(−1e9)

        exp_sim = torch.exp(sim)
        # Numerator: positive pairs
        pos = torch.exp(sim[torch.arange(2 * B), labels]).clamp(min=self.eps)
        denom = exp_sim.sum(dim=1).clamp(min=self.eps)

        loss = -(torch.log(pos / denom)).mean()
        return loss