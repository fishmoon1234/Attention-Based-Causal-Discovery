"""Argument groups, naming and per-epoch monitoring shared by the two trainers and the evaluator."""
import csv
import os

import numpy as np
import torch

from .data import LEVELS, Shard, shard_path
from .utils import ACYCLICITY, RelativeError, reconstruction_loss


# ----------------------------------------------------------------------------- arguments
def add_data_args(p):
    g = p.add_argument_group("data (must match a shard written by generate_data.py)")
    g.add_argument("--data_root", default="data")
    g.add_argument("--level", default="hetero", choices=LEVELS)
    g.add_argument("--graph_type", default="ER")
    g.add_argument("--s0", type=int, default=1)
    g.add_argument("--d", type=int, required=True, help="number of variables")
    g.add_argument("--M", type=int, default=101000, help="total domains in the shard")
    g.add_argument("--data_seed", type=int, default=42, help="seed the shard was generated with")
    g.add_argument("--n_test", type=int, default=1000, help="held-out domains (the last n_test)")
    g.add_argument("--noise_seed", type=int, default=0, help="seed of the synthesised observations")


def add_model_args(p):
    g = p.add_argument_group("model (paper defaults: r=1, k=20, L=35)")
    g.add_argument("--n_heads", type=int, default=1, help="attention heads r")
    g.add_argument("--dk", type=int, default=20, help="query/key width k")
    g.add_argument("--n_layers", type=int, default=35, help="attention layers L")
    g.add_argument("--n", type=int, default=1000, help="samples per domain (token length)")


def add_train_args(p):
    g = p.add_argument_group("training")
    g.add_argument("--M_train", type=int, default=None, help="training domains (default: all but the test set)")
    g.add_argument("--batch_size", type=int, default=100)
    g.add_argument("--weight_decay", type=float, default=1e-6)
    g.add_argument("--acyclicity", default="expm", choices=list(ACYCLICITY), help="form of h(A)")
    g.add_argument("--resample_noise", action="store_true",
                   help="draw fresh observations from the same SEMs every epoch")
    g.add_argument("--eval_train_domains", type=int, default=1000,
                   help="training domains used for the per-epoch monitoring metrics")
    g.add_argument("--seed", type=int, default=42, help="model initialisation / shuffling seed")
    g.add_argument("--init_ckpt", default=None, help="start from this state_dict instead of random init")
    g.add_argument("--out_dir", default="checkpoints")
    g.add_argument("--name", default=None, help="checkpoint stem (default: built from the arguments)")


def model_name(a):
    return (f"adag_{a.level}_{a.graph_type}{a.s0}_d{a.d}_n{a.n}_M{a.M_train}"
            f"_L{a.n_layers}_k{a.dk}_r{a.n_heads}_seed{a.seed}")


def load_shard(a, device):
    """Shard + (train, test) indices from the data arguments. Sets a.M_train."""
    path = shard_path(a.data_root, a.level, a.graph_type, a.s0, a.d, a.M, a.data_seed)
    shard = Shard(path, n=a.n, device=device, noise_seed=a.noise_seed,
                  resample_noise=getattr(a, "resample_noise", False))
    train, test = shard.split(a.n_test)
    if a.M_train is None:
        a.M_train = len(train)
    if a.M_train > len(train):
        raise SystemExit(f"--M_train {a.M_train} exceeds the {len(train)} non-test domains of {shard}")
    return shard, train[:a.M_train], test


def to_model_input(X):
    """Shard batches are [B, n, d]; the model wants [B, d, n]."""
    return X.permute(0, 2, 1).contiguous()


# ----------------------------------------------------------------------------- monitoring
@torch.no_grad()
def epoch_eval(model, shard, idx, n, batch_size, h_fn, max_domains=None):
    """Mean reconstruction loss, h(A), h(A)^2 and relative error of A over the given domains."""
    rel = RelativeError(reduction="sum")
    if max_domains is not None and len(idx) > max_domains:
        idx = idx[:max_domains]
    model.eval()
    tot = dict(rec=0.0, h=0.0, h2=0.0, rel=0.0)
    for s in range(0, len(idx), batch_size):
        X, W = shard.batch(idx[s:s + batch_size], n=n)
        x = to_model_input(X)
        x_hat, A = model(x)
        h = h_fn(A.transpose(1, 2))
        tot["rec"] += 0.5 * ((x_hat - x) ** 2).sum().item() / n
        tot["h"] += h.sum().item()
        tot["h2"] += (h ** 2).sum().item()
        tot["rel"] += rel(A.transpose(1, 2), W).item()
    return {k: v / len(idx) for k, v in tot.items()}


class CSVLogger:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])
