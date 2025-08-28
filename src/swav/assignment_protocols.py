import torch

@torch.no_grad()
def sinkhorn(out, n_iters=3, epsilon=0.03):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_of_cols
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

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