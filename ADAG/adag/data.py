"""
Synthetic multi-domain linear-SEM data: generation and storage-efficient loading.

Only the weighted adjacency matrices are stored (sparse, float32, in topological order);
the observations X of a domain are synthesised on the fly from the closed form of the
linear SEM

    X = X W + Z    <=>    X = Z (I - W)^{-1},    Z ~ noise

so a dataset of 100k domains with d = 50 is a few tens of MB instead of hundreds of GB,
and the per-domain sample size n is a *loading* argument rather than a property of the
data. W is held canonicalised into topological order, so (I - W)^T is triangular and X
costs a single triangular solve per batch.

Three data types (paper, Section 6, "Dataset Settings"), selected with ``level``:

  hetero        all domains share one DAG, edge weights are re-drawn per domain
                (Lu & Gao, 2023)                              -- structure "shared"
  orderconsist  all domains share one topological order, edge sets and weights vary
                (Chen et al., 2021)                           -- structure "shared_order"
  general       every domain draws an independent random DAG  -- structure "per_domain"

Layout of one dataset ("shard"):

    {root}/{level}/{graph_type}{s0}_d{d}_M{M}_seed{seed}/
        manifest.json    # every parameter needed to reproduce it
        W.npz            # sparse weights: val (f32), row / col (i16), order (i16), indptr

``M`` counts *all* domains in the shard; the last ``n_test`` of them are held out by
``Shard.split`` and never used for training.
"""
import json
import os

import numpy as np
import torch

LEVELS = ("hetero", "orderconsist", "general")
_LEVEL_ALIAS = {"full": "general", "fullDAG": "general", "sameDAG": "hetero"}
STRUCTURE = {"hetero": "shared", "orderconsist": "shared_order", "general": "per_domain"}
NOISE_TYPES = ("gauss", "exp", "gumbel", "uniform")


def shard_path(root, level, graph_type, s0, d, M, seed):
    """Canonical directory of one configuration."""
    level = _LEVEL_ALIAS.get(level, level)
    return os.path.join(root, level, f"{graph_type}{s0}_d{d}_M{M}_seed{seed}")


# ============================================================================= generation
def sample_dag(d, s0, graph_type, rng):
    """Random DAG with (about) s0 edges as a binary [d, d] matrix, B[i, j] = 1 <=> i -> j.

    ER: an Erdos-Renyi G(d, m=s0) graph with a random acyclic orientation; SF: a
    directed Barabasi-Albert graph with m = s0 / d edges per node. Mirrors
    ``notears.utils.simulate_dag``.
    """
    import igraph as ig
    if graph_type == "ER":
        G = ig.Graph.Erdos_Renyi(n=d, m=int(s0))
        B = np.array(G.get_adjacency().data, dtype=float)
        P = rng.permutation(np.eye(d))
        B = np.tril(P.T @ B @ P, k=-1)
    elif graph_type == "SF":
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = np.array(G.get_adjacency().data, dtype=float)
    else:
        raise ValueError(f"unknown graph_type {graph_type!r} (use ER or SF)")
    P = rng.permutation(np.eye(d))
    B = P.T @ B @ P
    assert ig.Graph.Adjacency(B.tolist()).is_dag()
    return B


def sample_weights(B, rng, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Edge weights drawn uniformly from the union of the ranges (notears convention)."""
    W = np.zeros_like(B, dtype=float)
    which = rng.integers(len(w_ranges), size=B.shape)
    for i, (lo, hi) in enumerate(w_ranges):
        W += B * (which == i) * rng.uniform(lo, hi, size=B.shape)
    return W


def canonicalise(W):
    """Permute W into a topological order so it is strictly upper triangular."""
    import networkx as nx
    order = np.array(list(nx.topological_sort(nx.DiGraph(np.asarray(W) != 0))), dtype=np.int64)
    Wc = W[np.ix_(order, order)]
    assert np.allclose(Wc, np.triu(Wc, 1)), "canonicalised W is not strictly upper triangular"
    return Wc, order


def generate_shard(level, graph_type, s0, d, M, seed, noise="gauss", noise_scale=1.0,
                   w_ranges=((-2.0, -0.5), (0.5, 2.0)), root="data", verbose=True):
    """Draw and write one dataset. Returns its directory.

    hetero:       one DAG from ``sample_dag``; weights re-drawn per domain on its edges.
    orderconsist: one random topological order; each domain picks s0*d edges uniformly
                  among the pairs consistent with that order (the MultiDAG scheme of
                  Chen et al., 2021, with the complete union support).
    general:      an independent ``sample_dag`` + weights per domain.
    """
    from tqdm import tqdm
    level = _LEVEL_ALIAS.get(level, level)
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if noise not in NOISE_TYPES:
        raise ValueError(f"noise must be one of {NOISE_TYPES}")
    n_edges = int(s0 * d)
    if n_edges > d * (d - 1) // 2:
        raise ValueError(f"{n_edges} edges requested but a DAG on {d} nodes holds at most "
                         f"{d * (d - 1) // 2}")
    rng = np.random.default_rng(seed)
    it = (lambda x, **k: tqdm(x, **k)) if verbose else (lambda x, **k: x)
    meta = dict(level=level, graph_type=graph_type, s0=int(s0), d=int(d), M=int(M),
                seed=int(seed), sem="linear", noise=noise, noise_scale=float(noise_scale),
                w_ranges=[list(r) for r in w_ranges], structure=STRUCTURE[level],
                nnz_per_domain=None, consistent_order=None, union_edges=None)

    if level == "hetero":
        Wc0, order = canonicalise(sample_weights(sample_dag(d, n_edges, graph_type, rng), rng))
        row, col = np.nonzero(Wc0)
        Bc = np.zeros((d, d)); Bc[row, col] = 1
        val = np.empty((M, row.shape[0]), dtype=np.float32)
        for m in it(range(M), desc="  drawing weights", unit="dom"):
            val[m] = sample_weights(Bc, rng, w_ranges)[row, col]
        arrays = dict(val=val.reshape(-1), row=row.astype(np.int16), col=col.astype(np.int16),
                      order=order.astype(np.int16))
        meta["nnz_per_domain"] = int(row.shape[0])

    elif level == "orderconsist":
        order = rng.permutation(d).astype(np.int64)       # canonical position -> variable
        iu, ju = np.triu_indices(d, k=1)                   # every pair allowed by the order
        rows, cols, vals = [], [], []
        for _ in it(range(M), desc="  drawing edges", unit="dom"):
            pick = rng.choice(iu.shape[0], size=n_edges, replace=False)
            r, c = iu[pick], ju[pick]
            B = np.zeros((d, d)); B[r, c] = 1
            w = sample_weights(B, rng, w_ranges)[r, c]
            rows.append(r.astype(np.int16)); cols.append(c.astype(np.int16)); vals.append(w.astype(np.float32))
        arrays = dict(indptr=np.arange(0, M * n_edges + 1, n_edges, dtype=np.int64),
                      row=np.concatenate(rows), col=np.concatenate(cols), val=np.concatenate(vals),
                      order=order.astype(np.int16))
        meta["consistent_order"] = order.tolist()
        meta["union_edges"] = int(iu.shape[0])

    else:  # general
        rows, cols, vals, orders, nnz = [], [], [], np.empty((M, d), np.int16), np.empty(M, np.int64)
        for m in it(range(M), desc="  drawing DAGs", unit="dom"):
            Wc, order = canonicalise(sample_weights(sample_dag(d, n_edges, graph_type, rng), rng, w_ranges))
            r, c = np.nonzero(Wc)
            rows.append(r.astype(np.int16)); cols.append(c.astype(np.int16))
            vals.append(Wc[r, c].astype(np.float32)); orders[m] = order; nnz[m] = r.shape[0]
        arrays = dict(indptr=np.concatenate([[0], np.cumsum(nnz)]).astype(np.int64),
                      row=np.concatenate(rows), col=np.concatenate(cols), val=np.concatenate(vals),
                      order=orders)

    out = shard_path(root, level, graph_type, s0, d, M, seed)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "W.npz"), **arrays)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    if verbose:
        size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
        print(f">> wrote {out}  ({size / 2**20:.1f} MB; storing X at n=1000 would be "
              f"{M * 1000 * d * 4 / 2**30:.1f} GB)")
    return out


# ============================================================================= loading
class Shard(torch.utils.data.Dataset):
    """One dataset. Indexes domains; ``batch(idx)`` returns (X, W) with X synthesised.

    W is returned in the ORIGINAL variable labelling with W[i, j] != 0 <=> i -> j.
    ``n`` is a constructor argument: any sample size can be drawn from the same shard.

    Args:
        path:            shard directory (see ``shard_path``).
        n:               samples per domain.
        device:          where X is synthesised ('cuda' recommended for training).
        noise_seed:      seeds the noise so a (domain, n) pair always yields the same X.
        resample_noise:  if True, ``set_epoch(e)`` changes the noise every epoch, turning
                         the fixed dataset into fresh draws from the same SEMs.
    """

    def __init__(self, path, n, device="cpu", noise_seed=0, resample_noise=False, dtype=torch.float32):
        self.path = path
        with open(os.path.join(path, "manifest.json")) as f:
            self.meta = json.load(f)
        self.meta["level"] = _LEVEL_ALIAS.get(self.meta["level"], self.meta["level"])
        z = np.load(os.path.join(path, "W.npz"))
        self.structure = self.meta["structure"]
        self.d, self.M, self.n = int(self.meta["d"]), int(self.meta["M"]), int(n)
        self.device, self.dtype = device, dtype
        self.noise_seed, self.resample_noise, self.epoch = int(noise_seed), bool(resample_noise), 0
        self.noise = self.meta.get("noise", self.meta.get("sem", "gauss"))
        if self.noise == "linear":
            self.noise = "gauss"
        self.noise_scale = float(self.meta.get("noise_scale", self.meta.get("noise_var", 1.0)))

        self.val = z["val"]
        self.row, self.col = z["row"].astype(np.int64), z["col"].astype(np.int64)
        self.order = z["order"].astype(np.int64)        # [d] or [M, d] for per_domain
        if self.structure == "shared":
            self.nnz = self.row.shape[0]
            self.val = self.val.reshape(self.M, self.nnz)
        else:
            self.indptr = z["indptr"].astype(np.int64)

    def __len__(self):
        return self.M

    def __repr__(self):
        m = self.meta
        return (f"Shard({m['level']} {m['graph_type']}{m['s0']} d={self.d} M={self.M} "
                f"seed={m['seed']} noise={self.noise} n={self.n})")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def split(self, n_test=1000):
        """Disjoint (train, test) domain indices: the last ``n_test`` domains are test."""
        assert self.M > n_test, f"M={self.M} too small for n_test={n_test}"
        return np.arange(0, self.M - n_test), np.arange(self.M - n_test, self.M)

    # ---- weights -----------------------------------------------------------
    def _canonical_W(self, idx):
        """Strictly upper-triangular W in topological order, plus that order. [B,d,d], [B,d]"""
        B, d = len(idx), self.d
        W = torch.zeros(B, d, d, dtype=self.dtype)
        if self.structure == "shared":
            W[:, self.row, self.col] = torch.from_numpy(np.asarray(self.val[idx], dtype=np.float32)).to(self.dtype)
            order = torch.from_numpy(self.order).expand(B, d)
        else:
            order = torch.empty(B, d, dtype=torch.long)
            for k, m in enumerate(idx):
                a, b = self.indptr[m], self.indptr[m + 1]
                W[k, self.row[a:b], self.col[a:b]] = torch.from_numpy(self.val[a:b].astype(np.float32)).to(self.dtype)
                order[k] = torch.from_numpy(self.order[m] if self.structure == "per_domain" else self.order)
        return W, order

    def weights(self, idx):
        """W in the original variable labelling. [B, d, d]"""
        Wc, order = self._canonical_W(np.asarray(idx))
        return self._uncanonicalise(Wc, order)

    @staticmethod
    def _uncanonicalise(Wc, order):
        B, d = Wc.shape[0], Wc.shape[1]
        inv = torch.argsort(order, dim=1)
        ar = torch.arange(B, device=Wc.device).unsqueeze(1)
        return Wc[ar.unsqueeze(2), inv.unsqueeze(2), inv.unsqueeze(1)]

    # ---- samples -----------------------------------------------------------
    def _noise(self, B, n, d, gen):
        kw = dict(device=self.device, dtype=self.dtype, generator=gen)
        if self.noise == "gauss":
            Z = torch.randn(B, n, d, **kw)
        elif self.noise == "exp":
            Z = -torch.log(torch.rand(B, n, d, **kw).clamp_min(1e-12))
        elif self.noise == "gumbel":
            Z = -torch.log(-torch.log(torch.rand(B, n, d, **kw).clamp(1e-12, 1 - 1e-7)))
        elif self.noise == "uniform":
            Z = 2 * torch.rand(B, n, d, **kw) - 1
        else:
            raise ValueError(self.noise)
        return Z * self.noise_scale if self.noise_scale != 1.0 else Z

    def batch(self, idx, n=None):
        """(X, W) for the given domain indices. X: [B, n, d], W: [B, d, d] (original order)."""
        idx = np.asarray(idx)
        n = int(n or self.n)
        Wc, order = self._canonical_W(idx)
        Wc, order = Wc.to(self.device), order.to(self.device)
        B, d = Wc.shape[0], self.d

        gen = torch.Generator(device=self.device)
        tag = self.epoch if self.resample_noise else 0
        gen.manual_seed((self.noise_seed * 1_000_003 + int(idx[0]) * 1009 + tag) % (2**63 - 1))
        Z = self._noise(B, n, d, gen)

        I = torch.eye(d, device=self.device, dtype=self.dtype).expand(B, d, d)
        A = (I - Wc).transpose(1, 2).contiguous()                    # lower triangular
        Xc = torch.linalg.solve_triangular(A, Z.transpose(1, 2), upper=False).transpose(1, 2)
        X = torch.empty_like(Xc)
        X.scatter_(2, order.unsqueeze(1).expand(B, n, d), Xc)       # back to original labels
        return X, self._uncanonicalise(Wc, order)

    def loader(self, indices, batch_size, shuffle=True, n=None, seed=None):
        """Yields (X [B,n,d], W [B,d,d]) batches over the given domain indices."""
        indices = np.asarray(indices)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(indices)) if shuffle else np.arange(len(indices))
        for i in range(0, len(indices), batch_size):
            yield self.batch(indices[perm[i:i + batch_size]], n=n)
