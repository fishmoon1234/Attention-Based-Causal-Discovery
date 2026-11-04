"""Losses, acyclicity constraints, metrics and small helpers shared by the scripts."""
import random

import numpy as np
import torch


# ----------------------------------------------------------------------------- misc
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def lr_at(epoch, lr0, step_size, gamma):
    """Step decay: lr0 * gamma ** (epoch // step_size)."""
    return lr0 * (gamma ** (epoch // step_size))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


# ----------------------------------------------------------------------------- losses
class RelativeError:
    """Per-sample relative L2 error ||x - y||_2 / ||y||_2 over flattened samples.

    ``reduction`` is 'mean', 'sum' or 'none'. Used as the supervised loss on A and as
    the reported 'relative error of A' everywhere.
    """

    def __init__(self, reduction="mean", eps=1e-12):
        self.reduction, self.eps = reduction, eps

    def __call__(self, x, y):
        b = x.shape[0]
        diff = torch.norm(x.reshape(b, -1) - y.reshape(b, -1), 2, dim=1)
        ref = torch.norm(y.reshape(b, -1), 2, dim=1) + self.eps
        r = diff / ref
        if self.reduction == "mean":
            return r.mean()
        if self.reduction == "sum":
            return r.sum()
        return r


def reconstruction_loss(x_hat, x, n):
    """0.5 * ||X_hat - X||_F^2 / n, averaged over the batch (paper Eq. 14)."""
    return 0.5 * ((x_hat - x) ** 2).sum() / (x.shape[0] * n)


# ----------------------------------------------------------------------------- acyclicity
def acyclicity_expm(A):
    """Polynomial form of the NOTEARS constraint h(A) = tr[(I + A∘A/d)^d] - d, per domain.

    Equals zero iff A is acyclic (Yu et al., 2019). A: [B, d, d] -> h: [B].
    """
    B, d = A.shape[0], A.shape[1]
    M = torch.eye(d, device=A.device, dtype=A.dtype).expand(B, d, d) + A * A / d
    E = torch.linalg.matrix_power(M, d - 1)
    return (E.transpose(1, 2) * M).sum(dim=(1, 2)) - d


def acyclicity_logdet(A, s=1.0, eps=1e-6):
    """DAGMA log-det constraint h(A) = -log det(sI - A∘A) + d log s, per domain."""
    B, d = A.shape[0], A.shape[1]
    I = torch.eye(d, device=A.device, dtype=A.dtype).expand(B, d, d)
    _, logabsdet = torch.linalg.slogdet(s * I - A * A + eps * I)
    return -logabsdet + d * torch.log(torch.tensor(float(s), device=A.device))


ACYCLICITY = {"expm": acyclicity_expm, "logdet": acyclicity_logdet}


# ----------------------------------------------------------------------------- metrics
def threshold_adjacency(W, w_threshold=0.3):
    """Binary adjacency from weights: |W| >= w_threshold, zero diagonal."""
    B = np.abs(np.asarray(W)) >= w_threshold
    np.fill_diagonal(B, False)
    return B


def is_dag(B):
    import networkx as nx
    return nx.is_directed_acyclic_graph(nx.DiGraph(np.asarray(B, dtype=int)))


def graph_metrics(W_est, W_true, w_threshold=0.3):
    """Structure and weight metrics for one domain.

    SHD counts reversed + extra + missing edges (an undirected edge counts both ways),
    on every domain whether or not the estimate is acyclic. precision / recall / F1 are
    on directed edges. AUROC uses the raw |W_est| as a score. rel_error is the relative
    Frobenius error of the raw weights.
    """
    from sklearn.metrics import roc_auc_score
    W_est, W_true = np.asarray(W_est, float), np.asarray(W_true, float)
    d = W_true.shape[0]
    T = W_true != 0
    B = threshold_adjacency(W_est, w_threshold)
    rev = B & T.T & ~T
    extra = B & ~T & ~T.T
    missing = T & ~B & ~B.T
    tp, fp, fn = int((B & T).sum()), int((B & ~T).sum()), int((~B & T).sum())
    off = ~np.eye(d, dtype=bool)
    y = T[off].astype(int)
    return dict(
        shd=int(rev.sum() + extra.sum() + missing.sum()),
        nnz=int(B.sum()),
        precision=tp / (tp + fp) if tp + fp else float("nan"),
        recall=tp / (tp + fn) if tp + fn else float("nan"),
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        auroc=roc_auc_score(y, np.abs(W_est[off])) if 0 < y.sum() < len(y) else float("nan"),
        rel_error=float(np.linalg.norm(W_est - W_true) / max(np.linalg.norm(W_true), 1e-12)),
        is_dag=bool(is_dag(B)),
    )


METRIC_KEYS = ["shd", "nnz", "precision", "recall", "f1", "auroc", "rel_error"]


def summarize(rows, keys=METRIC_KEYS):
    """mean/std over a list of graph_metrics dicts, ignoring NaNs."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)))
    out["dag_frac"] = float(np.mean([r["is_dag"] for r in rows]))
    return out


def format_summary(summ, keys=METRIC_KEYS):
    lines = [f"  {k:<10} {summ[k]['mean']:9.4f} +/- {summ[k]['std']:.4f}" for k in keys]
    lines.append(f"  {'acyclic':<10} {100 * summ['dag_frac']:8.1f}%")
    return "\n".join(lines)
