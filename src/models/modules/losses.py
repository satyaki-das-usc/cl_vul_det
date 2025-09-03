import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

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