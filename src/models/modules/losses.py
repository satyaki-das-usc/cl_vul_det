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

class OrthogonalProjectionLoss(torch.nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma  # scales the inter-class term

    def forward(self, features: torch.Tensor, targets: torch.Tensor):
        """
        features: tensor of shape (batch_size, feature_dim)
        targets: tensor of shape (batch_size,) with class indices
        """
        # Normalize features for cosine similarity
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)

        # Create a similarity matrix: (batch_size, batch_size)
        sim_matrix = features @ features.t()

        # Masks for same-class and different-class pairs
        targets = targets.view(-1, 1)
        same_mask = targets == targets.t()
        diff_mask = ~same_mask

        # Exclude self-similarity for same-class computations
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        same_mask = same_mask & ~diag_mask

        # Compute mean similarities
        if same_mask.sum() > 0:
            s = sim_matrix[same_mask].mean()
        else:
            s = torch.tensor(1.0, device=features.device)

        if diff_mask.sum() > 0:
            d = sim_matrix[diff_mask].mean()
        else:
            d = torch.tensor(0.0, device=features.device)

        loss_opl = (1 - s) + self.gamma * d.abs()
        return loss_opl