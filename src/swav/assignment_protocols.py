import torch

def sinkhorn(out, n_iters=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t()
    Q /= Q.sum()
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= Q.sum(dim=0, keepdim=True)
    return (Q / Q.sum(dim=0, keepdim=True)).t()

def uot_sinkhorn_gpu(scores: torch.Tensor,
                     epsilon=0.05,
                     rho=0.5,
                     n_iters=5):
    """
    GPU-only, batched unbalanced Sinkhorn-like updates.
    scores: [B, K] = z @ prototypes
    Returns Q: [B, K] soft assignment distributions.
    """
    Q = torch.exp(scores / epsilon)  # non-negative weights [B,K]
    for _ in range(n_iters):
        # Row update with relaxation: softly approach uniform mass
        row_sum = Q.sum(dim=1, keepdim=True)
        Q = Q / (row_sum + rho)
        # Column update with relaxation
        col_sum = Q.sum(dim=0, keepdim=True)
        Q = Q / (col_sum + rho)
    # Normalize rows to sum to one (softmax-like)
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
    return Q

if __name__ == "__main__":
    pass