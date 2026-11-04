"""
Unsupervised training of ADAG with the augmented Lagrangian method (the method of the
paper; Section 3, Eq. 7, and Appendix A, Algorithm 1).

No ground-truth graphs are used. For training domains X^1..X^M the objective is

    min_Theta  sum_eta  0.5 ||A(X^eta) X^eta - X^eta||^2 / n           (reconstruction)
             + alpha * mean_eta h(A(X^eta)) + rho/2 * mean_eta h(A(X^eta))^2   (acyclicity)
             + lamb_l1 * ||A(X^eta)||_1                                  (sparsity)

with h(A) = tr[(I + A∘A/d)^d] - d (= 0 iff A is a DAG). Training has two stages:

  warmup  alpha = alpha0 (0) and rho = rho0 (1) fixed, --warmup_epochs epochs with the
          learning rate decaying by --gamma every --step_size epochs       (paper: 5000 epochs,
          lr 3e-4, decay 0.7 / 1000 epochs)
  alm     Algorithm 1: repeat { train --inner_epochs epochs at --inner_lr; if h did not
          drop to --c of its previous value, rho *= 10 and repeat; alpha += rho * h }
          until mean h < --h_tol or --max_outer iterations                 (paper: 100 epochs,
          lr 1e-4, c = 1/4, h_tol = 1e-5)

  python train_adag_alm.py --data_root data --level general --s0 1 --d 5 --n 1000 --M_train 100000
  python train_adag_alm.py ... --stage alm --init_ckpt checkpoints/<name>_warmup.ckpt   # resume

A single multiplier alpha is shared by all domains (the per-domain multipliers of Eq. 14
are updated with the batch-mean of h, which is what the released results used).

Checkpoints in --out_dir:  {name}_init.ckpt (initialisation), {name}_warmup.ckpt (end of
warm-up), {name}_alm.ckpt (final ALM iterate, what Algorithm 1 returns) and
{name}_alm_best.ckpt (best by --select_by on the test domains). Ground-truth relative
error of A is printed for monitoring only; it enters nothing unless --select_by rel_error.
"""
import argparse, math, os, time

import torch

from adag import ADAG
from adag.training import (CSVLogger, add_data_args, add_model_args, add_train_args, epoch_eval,
                           load_shard, model_name, to_model_input)
from adag.utils import ACYCLICITY, count_params, lr_at, reconstruction_loss, set_lr, set_seed


def train_epochs(model, opt, shard, train_idx, test_idx, a, h_fn, alpha, rho, epochs, lr_fn, log,
                 stage, state):
    """Minimise the augmented Lagrangian at fixed (alpha, rho) for `epochs` epochs."""
    for ep in range(epochs):
        t0 = time.time()
        lr = set_lr(opt, lr_fn(ep))
        state["epoch"] += 1
        shard.set_epoch(state["epoch"])
        model.train()
        for X, _ in shard.loader(train_idx, a.batch_size, shuffle=True, seed=a.seed + state["epoch"]):
            x = to_model_input(X)
            opt.zero_grad()
            x_hat, A = model(x)
            h = torch.clamp(torch.nan_to_num(h_fn(A.transpose(1, 2)), nan=0.0), max=1e8)
            loss = (reconstruction_loss(x_hat, x, a.n) + alpha * h.mean() + 0.5 * rho * (h ** 2).mean()
                    + a.lamb_l1 * A.abs().sum() / x.shape[0])
            loss.backward()
            opt.step()

        tr = epoch_eval(model, shard, train_idx, a.n, a.batch_size, h_fn, a.eval_train_domains)
        te = epoch_eval(model, shard, test_idx, a.n, a.batch_size, h_fn)
        crit = te["rel"] if a.select_by == "rel_error" else te["rec"]
        if crit < state["best"]:
            state["best"] = crit
            torch.save(model.state_dict(), f"{a.out_dir}/{a.name}_{stage}_best.ckpt")
        dt = time.time() - t0
        log.log(dict(stage=stage, outer=state["outer"], epoch=state["epoch"], time=f"{dt:.1f}", lr=f"{lr:.2e}",
                     alpha=alpha, rho=rho, train_rec=tr["rec"], train_h=tr["h"], train_rel=tr["rel"],
                     test_rec=te["rec"], test_h=te["h"], test_rel=te["rel"]))
        print(f"[{stage} outer {state['outer']:2d}] epoch {ep:4d}/{epochs} {dt:6.1f}s lr {lr:.1e} "
              f"alpha {alpha:.3g} rho {rho:.3g} | train rec {tr['rec']:.4f} h {tr['h']:.2e} relA {tr['rel']:.4f} "
              f"| test rec {te['rec']:.4f} h {te['h']:.2e} relA {te['rel']:.4f}", flush=True)
    return tr["h"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p); add_model_args(p); add_train_args(p)
    g = p.add_argument_group("augmented Lagrangian")
    g.add_argument("--stage", default="both", choices=["both", "warmup", "alm"])
    g.add_argument("--alpha0", type=float, default=0.0, help="initial Lagrange multiplier")
    g.add_argument("--rho0", type=float, default=1.0, help="initial penalty weight")
    g.add_argument("--c", type=float, default=0.25, help="required progress on h per outer iteration")
    g.add_argument("--h_tol", type=float, default=1e-5, help="stop when mean h(A) on training domains < h_tol")
    g.add_argument("--rho_max", type=float, default=1e16)
    g.add_argument("--max_outer", type=int, default=100, help="maximum outer iterations T_max")
    g.add_argument("--lamb_l1", type=float, default=0.02, help="L1 sparsity weight on A")
    g.add_argument("--warmup_epochs", type=int, default=5000)
    g.add_argument("--lr", type=float, default=3e-4, help="warm-up learning rate")
    g.add_argument("--gamma", type=float, default=0.7, help="warm-up lr decay factor")
    g.add_argument("--step_size", type=int, default=1000, help="warm-up epochs between decays")
    g.add_argument("--inner_epochs", type=int, default=100, help="epochs per outer ALM iteration")
    g.add_argument("--inner_lr", type=float, default=1e-4, help="ALM learning rate (scaled down by log10 rho)")
    g.add_argument("--select_by", default="rec", choices=["rec", "rel_error"],
                   help="criterion for *_best.ckpt: test reconstruction (unsupervised) or ground-truth error")
    a = p.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard, train_idx, test_idx = load_shard(a, device)
    a.name = a.name or model_name(a)
    os.makedirs(a.out_dir, exist_ok=True)
    model = ADAG(a.n_heads, a.dk, a.n_layers, a.d, a.n).to(device)
    if a.init_ckpt:
        model.load_state_dict(torch.load(a.init_ckpt, map_location=device))
    elif a.stage == "alm" and os.path.exists(f"{a.out_dir}/{a.name}_warmup.ckpt"):
        a.init_ckpt = f"{a.out_dir}/{a.name}_warmup.ckpt"
        model.load_state_dict(torch.load(a.init_ckpt, map_location=device))
    torch.save(model.state_dict(), f"{a.out_dir}/{a.name}_init.ckpt")
    print(f">> {shard}\n>> device {device}   train domains {len(train_idx)}   test domains {len(test_idx)}")
    print(f">> ADAG r={a.n_heads} k={a.dk} L={a.n_layers} d={a.d} n={a.n}: {count_params(model):,} parameters"
          + (f"   init from {a.init_ckpt}" if a.init_ckpt else ""))
    print(f">> stage {a.stage}; checkpoints -> {a.out_dir}/{a.name}_*.ckpt")

    h_fn = ACYCLICITY[a.acyclicity]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    log = CSVLogger(f"{a.out_dir}/{a.name}_alm_log.csv",
                    ["stage", "outer", "epoch", "time", "lr", "alpha", "rho", "train_rec", "train_h",
                     "train_rel", "test_rec", "test_h", "test_rel"])
    state = dict(epoch=-1, outer=0, best=float("inf"))
    alpha, rho = a.alpha0, a.rho0

    if a.stage in ("both", "warmup"):
        print(f"********** warm-up: alpha={alpha}, rho={rho}, {a.warmup_epochs} epochs **********")
        train_epochs(model, opt, shard, train_idx, test_idx, a, h_fn, alpha, rho, a.warmup_epochs,
                     lambda ep: lr_at(ep, a.lr, a.step_size, a.gamma), log, "warmup", state)
        torch.save(model.state_dict(), f"{a.out_dir}/{a.name}_warmup.ckpt")
        print(f"********** warm-up done -> {a.out_dir}/{a.name}_warmup.ckpt **********")

    if a.stage in ("both", "alm"):
        print(f"********** augmented Lagrangian (Algorithm 1) **********")
        state["best"] = float("inf")
        h_old = epoch_eval(model, shard, train_idx, a.n, a.batch_size, h_fn, a.eval_train_domains)["h"]
        print(f">> initial mean h(A) on training domains: {h_old:.3e}")
        for outer in range(1, a.max_outer + 1):
            state["outer"] = outer
            while rho < a.rho_max:
                lr_fn = lambda ep, r=rho: max(min(a.inner_lr, a.inner_lr / max(math.log10(r), 1e-10)), 1e-8)
                print(f">> outer {outer}: alpha={alpha:.4g} rho={rho:.4g} lr={lr_fn(0):.2e}")
                h_new = train_epochs(model, opt, shard, train_idx, test_idx, a, h_fn, alpha, rho,
                                     a.inner_epochs, lr_fn, log, "alm", state)
                if h_new > a.c * h_old:
                    rho *= 10
                else:
                    break
            alpha += rho * h_new
            h_old = h_new
            torch.save(model.state_dict(), f"{a.out_dir}/{a.name}_alm.ckpt")
            print(f">> outer {outer} done: mean h(A) = {h_new:.3e}  (alpha -> {alpha:.4g}, rho = {rho:.4g})")
            if h_new <= a.h_tol:
                print(f">> converged: h < {a.h_tol}")
                break
            if rho >= a.rho_max:
                print(">> stopping: rho reached rho_max")
                break
        print(f"********** ALM done -> {a.out_dir}/{a.name}_alm.ckpt (final) and "
              f"{a.name}_alm_best.ckpt (best test {a.select_by}) **********")


if __name__ == "__main__":
    main()
