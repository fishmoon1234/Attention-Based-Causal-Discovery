"""
The ADAG model: a linear-attention network that maps the observations of one domain,
X in R^{d x n}, to the weighted adjacency matrix A in R^{d x d} of its linear SEM
X = A^T X + E  (paper, Section 3, Eqs. 2-5).

Tokens are variables: the i-th token is the length-n vector of observations of X_i.

    H_0      = LayerNorm(X)                                   [d, n]
    H_l      = LayerNorm(sum_r Q_l^r (K_l^r)^T H_{l-1} / sqrt(k)) + H_{l-1},  l = 1..L-1
    A        = sum_r Q_L^r (K_L^r)^T / sqrt(k)   with the diagonal masked to zero
    X_hat    = A X                                            (reconstruction, Eq. 6)

where Q_l^r = H_{l-1} W_l^{Q,r}, K_l^r = H_{l-1} W_l^{K,r} and W^{Q,r}, W^{K,r} in R^{n x k}.
Every attention layer is linear in H (no softmax), which is what lets the layer-wise update
be read as a fixed-point iteration for the DAG (paper, Section 4).

The model returns ``(X_hat, A)``; ``A[b, i, j]`` is the estimated weight of the edge
j -> i in the row-convention of ``X_hat = A X``.  All training and evaluation code in
this repository compares ``A.transpose(1, 2)`` with the ground-truth matrix W, where
``W[i, j] != 0`` means i -> j (the NOTEARS / DAGMA convention X = X W + Z).
"""
import torch
import torch.nn as nn


class ADAG(nn.Module):
    """Linear-attention DAG estimator.

    Args:
        n_heads:   number of attention heads r (the paper uses 1).
        dk:        key/query width k (the paper uses 20).
        n_layers:  number of attention layers L (the paper uses 35).
        d:         number of variables (tokens).
        n:         samples per domain, i.e. the token (feature) length. A trained model
                   is tied to this n; see the README for using a model at a different n.

    Input:  x of shape [batch, d, n].
    Output: (x_hat [batch, d, n], A [batch, d, d]).

    Parameter names (``fcq``, ``fck``, ``fcn``, ``fcpx``) are kept from the original
    implementation so that released checkpoints load with ``strict=True``. ``fcpx`` is a
    legacy per-head d x d projection that the paper's formulation (Eq. 5) does not use;
    it is still registered so that checkpoint keys match, but it does not enter ``forward``.
    """

    def __init__(self, n_heads, dk, n_layers, d, n):
        super().__init__()
        self.n_heads = int(n_heads)
        self.dk = int(dk)
        self.n_layers = int(n_layers)
        self.d = int(d)
        self.n = int(n)
        self.scale = 1.0 / (self.dk ** 0.5)

        # query / key maps W^{Q,r}_l, W^{K,r}_l : R^n -> R^k, one per layer and head
        self.fcq = nn.ModuleList([nn.ModuleList([nn.Linear(self.n, self.dk) for _ in range(self.n_heads)])
                                  for _ in range(self.n_layers)])
        self.fck = nn.ModuleList([nn.ModuleList([nn.Linear(self.n, self.dk) for _ in range(self.n_heads)])
                                  for _ in range(self.n_layers)])
        # layer normalisation over the whole [d, n] token block, one per layer
        self.fcn = nn.ModuleList([nn.LayerNorm([self.d, self.n]) for _ in range(self.n_layers)])
        # legacy, unused in forward (see class docstring)
        self.fcpx = nn.ModuleList([nn.Linear(self.d, self.d, bias=False) for _ in range(self.n_heads)])

        # zero diagonal: a variable is never its own parent
        self.register_buffer("mask", torch.ones(self.d, self.d) - torch.eye(self.d), persistent=False)

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.d or x.shape[2] != self.n:
            raise ValueError(f"expected input [batch, d={self.d}, n={self.n}], got {tuple(x.shape)}")
        h = self.fcn[0](x)                                           # [B, d, n]

        # intermediate linear-attention layers with residual connection
        for l in range(self.n_layers - 1):
            upd = torch.zeros_like(h)
            for r in range(self.n_heads):
                q = self.fcq[l][r](h)                                # [B, d, k]
                k = self.fck[l][r](h)                                # [B, d, k]
                kv = torch.matmul(k.transpose(1, 2), h) * self.scale  # [B, k, n]
                upd = upd + torch.matmul(q, kv)                      # [B, d, n]
            h = self.fcn[l + 1](upd) + h

        # output layer: the attention matrix itself is the weighted adjacency estimate
        A = torch.zeros(x.shape[0], self.d, self.d, device=x.device, dtype=x.dtype)
        for r in range(self.n_heads):
            q = self.fcq[-1][r](h)
            k = self.fck[-1][r](h)
            A = A + torch.matmul(q, k.transpose(1, 2)) * self.scale * self.mask
        x_hat = torch.matmul(A, x)                                   # [B, d, n]
        return x_hat, A

    def estimate(self, x):
        """Weighted adjacency in the W convention (W[i, j] != 0  <=>  i -> j). [B, d, d]"""
        return self.forward(x)[1].transpose(1, 2)

    @classmethod
    def from_checkpoint(cls, path, n_heads, dk, n_layers, d, n, device="cpu"):
        model = cls(n_heads, dk, n_layers, d, n)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        return model.to(device)
