"""
Zero-shot evaluation of a trained ADAG model on the held-out domains of a dataset.

The model is applied once per test domain (a single forward pass, no optimisation) and the
estimate is thresholded at |w| >= --w_threshold (0.3, the standard protocol of Zheng et
al., 2018). Per-domain SHD, number of edges, precision, recall, F1, AUROC and relative
error of the weights are written to --out_dir, with a JSON summary and the raw estimates.

  python evaluate_adag.py --ckpt demo/checkpoints/adag_d5_n1000_general.ckpt \
      --data_root demo/data --level hetero --s0 1 --d 5 --n 1000

The architecture flags (--n_heads --dk --n_layers --d --n) must match the checkpoint. If
the test domains have fewer than n samples, pass --resample_to_n to pad each domain to n
by resampling its observations with replacement (paper, Appendix A).
"""
import argparse, json, os, time

import numpy as np
import torch

from adag import ADAG, Shard, shard_path
from adag.utils import METRIC_KEYS, format_summary, graph_metrics, summarize


def add_data_args(p):
    p.add_argument("--data_root", default="data")
    p.add_argument("--level", default="hetero", choices=["hetero", "orderconsist", "general"])
    p.add_argument("--graph_type", default="ER")
    p.add_argument("--s0", type=int, default=1)
    p.add_argument("--d", type=int, required=True, help="number of variables")
    p.add_argument("--M", type=int, default=101000, help="total domains in the shard")
    p.add_argument("--data_seed", type=int, default=42, help="seed the shard was generated with")
    p.add_argument("--n_test", type=int, default=1000, help="held-out domains (the last n_test)")
    p.add_argument("--noise_seed", type=int, default=0, help="seed of the synthesised observations")


def add_model_args(p):
    p.add_argument("--n_heads", type=int, default=1, help="attention heads r")
    p.add_argument("--dk", type=int, default=20, help="query/key width k")
    p.add_argument("--n_layers", type=int, default=35, help="attention layers L")
    p.add_argument("--n", type=int, default=1000, help="samples per domain the model was trained with")


def evaluate(model, shard, test_idx, n, batch_size=100, w_threshold=0.3, n_available=None):
    """Returns (per-domain metric rows, A_est [T, d, d], wall seconds)."""
    device = next(model.parameters()).device
    rows, A_all = [], np.zeros((len(test_idx), shard.d, shard.d), dtype=np.float32)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(test_idx), batch_size):
            idx = test_idx[s:s + batch_size]
            X, W = shard.batch(idx, n=n_available or n)                    # X [B, n_avail, d]
            if n_available and n_available < n:                            # pad by resampling
                pick = torch.randint(0, n_available, (n,), device=X.device)
                X = X[:, pick, :]
            W_est = model.estimate(X.permute(0, 2, 1).contiguous())        # [B, d, d], W convention
            W_est, W = W_est.cpu().numpy(), W.cpu().numpy()
            for k in range(len(idx)):
                m = graph_metrics(W_est[k], W[k], w_threshold); m["domain"] = int(idx[k])
                rows.append(m); A_all[s + k] = W_est[k]
    return rows, A_all, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    add_data_args(p); add_model_args(p)
    p.add_argument("--resample_to_n", type=int, default=None,
                   help="samples actually available per test domain; padded to --n by resampling")
    p.add_argument("--w_threshold", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--out_dir", default="results/adag")
    p.add_argument("--tag", default=None, help="output name (default: built from the arguments)")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = shard_path(a.data_root, a.level, a.graph_type, a.s0, a.d, a.M, a.data_seed)
    shard = Shard(path, n=a.n, device=device, noise_seed=a.noise_seed)
    _, test = shard.split(a.n_test)
    model = ADAG.from_checkpoint(a.ckpt, a.n_heads, a.dk, a.n_layers, a.d, a.n, device=device)
    print(f">> {shard}\n>> {os.path.basename(a.ckpt)} on {device}  test domains={len(test)}  "
          f"threshold={a.w_threshold}")

    rows, A_all, wall = evaluate(model, shard, test, a.n, a.batch_size, a.w_threshold, a.resample_to_n)
    summ = summarize(rows)
    summ.update(ckpt=a.ckpt, data=path, n=a.n, n_available=a.resample_to_n or a.n,
                w_threshold=a.w_threshold, domains=len(test), wall_clock_s=wall,
                seconds_per_domain=wall / len(test))

    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or f"{a.level}_{a.graph_type}{a.s0}_d{a.d}_n{a.n}_test{a.n_test}"
    np.save(os.path.join(a.out_dir, f"A_est_{tag}.npy"), A_all)
    np.savetxt(os.path.join(a.out_dir, f"metrics_{tag}.csv"),
               np.array([[r[k] for k in METRIC_KEYS] + [r["domain"]] for r in rows]),
               delimiter=",", header=",".join(METRIC_KEYS + ["domain"]), comments="")
    with open(os.path.join(a.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n---------- ADAG  {a.level} {a.graph_type}{a.s0} d={a.d} n={a.n} ----------")
    print(format_summary(summ))
    print(f"  {'time/domain':<10} {1e3 * wall / len(test):8.2f} ms")
    print(f">> written to {a.out_dir}/*_{tag}.*")


if __name__ == "__main__":
    main()
