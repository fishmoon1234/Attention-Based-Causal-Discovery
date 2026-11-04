"""
Supervised training of ADAG.

This variant uses the ground-truth weighted adjacency matrices of the training domains:

    loss = RelErr(A(X), W)  +  lamb_rec * 0.5 ||A(X) X - X||^2 / n
           [+ alpha * h(A) + rho/2 * h(A)^2, both off by default]

It is NOT the method of the paper, which is unsupervised (see train_adag_alm.py); it is
provided because ground-truth supervision is a natural reference point and makes a
useful initialisation / ablation. The architecture, data pipeline, checkpoint format and
evaluation are identical to the unsupervised trainer, so the two are directly comparable.

  python train_adag_supervised.py --data_root data --level hetero --s0 1 --d 5 --n 1000 \
      --M_train 100000 --epochs 1000

Per epoch, the reconstruction loss, h(A) and the relative error of A are reported on a
subset of the training domains and on the held-out test domains; the checkpoint with the
lowest test relative error is kept as  {out_dir}/{name}_supervised.ckpt  (the last epoch
as {name}_supervised_last.ckpt) and the curves go to {name}_supervised_log.csv.
"""
import argparse, os, time

import numpy as np
import torch

from adag import ADAG
from adag.training import (CSVLogger, add_data_args, add_model_args, add_train_args, epoch_eval,
                           load_shard, model_name, to_model_input)
from adag.utils import ACYCLICITY, RelativeError, count_params, lr_at, reconstruction_loss, set_lr, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p); add_model_args(p); add_train_args(p)
    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=1000)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--gamma", type=float, default=0.7, help="lr decay factor")
    g.add_argument("--step_size", type=int, default=300, help="epochs between lr decays")
    g.add_argument("--lamb_rec", type=float, default=1.0, help="weight of the reconstruction loss")
    g.add_argument("--alpha", type=float, default=0.0, help="linear acyclicity penalty (off by default)")
    g.add_argument("--rho", type=float, default=0.0, help="quadratic acyclicity penalty (off by default)")
    a = p.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard, train_idx, test_idx = load_shard(a, device)
    name = a.name or model_name(a)
    os.makedirs(a.out_dir, exist_ok=True)
    model = ADAG(a.n_heads, a.dk, a.n_layers, a.d, a.n).to(device)
    if a.init_ckpt:
        model.load_state_dict(torch.load(a.init_ckpt, map_location=device))
    torch.save(model.state_dict(), f"{a.out_dir}/{name}_init.ckpt")
    print(f">> {shard}\n>> device {device}   train domains {len(train_idx)}   test domains {len(test_idx)}")
    print(f">> ADAG r={a.n_heads} k={a.dk} L={a.n_layers} d={a.d} n={a.n}: {count_params(model):,} parameters")
    print(f">> checkpoints -> {a.out_dir}/{name}_supervised*.ckpt")

    h_fn = ACYCLICITY[a.acyclicity]
    rel = RelativeError(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    log = CSVLogger(f"{a.out_dir}/{name}_supervised_log.csv",
                    ["epoch", "time", "lr", "train_rec", "train_h", "train_rel", "test_rec", "test_h", "test_rel"])
    best = float("inf")
    for ep in range(a.epochs):
        t0 = time.time()
        lr = set_lr(opt, lr_at(ep, a.lr, a.step_size, a.gamma))
        shard.set_epoch(ep)
        model.train()
        for X, W in shard.loader(train_idx, a.batch_size, shuffle=True, seed=a.seed + ep):
            x = to_model_input(X)
            opt.zero_grad()
            x_hat, A = model(x)
            At = A.transpose(1, 2)
            loss = rel(At, W) + a.lamb_rec * reconstruction_loss(x_hat, x, a.n)
            if a.alpha or a.rho:
                h = torch.nan_to_num(h_fn(At), nan=0.0, posinf=1e8, neginf=-1e8)
                loss = loss + a.alpha * h.mean() + 0.5 * a.rho * (h ** 2).mean()
            loss.backward()
            opt.step()

        tr = epoch_eval(model, shard, train_idx, a.n, a.batch_size, h_fn, a.eval_train_domains)
        te = epoch_eval(model, shard, test_idx, a.n, a.batch_size, h_fn)
        dt = time.time() - t0
        if te["rel"] < best:
            best = te["rel"]
            torch.save(model.state_dict(), f"{a.out_dir}/{name}_supervised.ckpt")
        torch.save(model.state_dict(), f"{a.out_dir}/{name}_supervised_last.ckpt")
        log.log(dict(epoch=ep, time=f"{dt:.1f}", lr=f"{lr:.2e}", train_rec=tr["rec"], train_h=tr["h"],
                     train_rel=tr["rel"], test_rec=te["rec"], test_h=te["h"], test_rel=te["rel"]))
        print(f"epoch {ep:4d}/{a.epochs}  {dt:6.1f}s  lr {lr:.1e} | train rec {tr['rec']:.4f} h {tr['h']:.4f} "
              f"relA {tr['rel']:.4f} | test rec {te['rec']:.4f} h {te['h']:.4f} relA {te['rel']:.4f} "
              f"| best test relA {best:.4f}", flush=True)
    print(f">> done. best test relative error {best:.4f}  ->  {a.out_dir}/{name}_supervised.ckpt")


if __name__ == "__main__":
    main()
