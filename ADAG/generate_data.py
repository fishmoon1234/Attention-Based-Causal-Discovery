"""
Generate multi-domain linear-SEM datasets for ADAG.

One script covers the three data types of the paper (Section 6):

  --level hetero        all domains share one DAG, weights differ      (Lu & Gao, 2023)
  --level orderconsist  all domains share one topological order        (Chen et al., 2021)
  --level general       every domain has an independent random DAG

Only the weighted adjacency matrices are written (see adag/data.py); observations are
synthesised at load time, so the per-domain sample size n is chosen when training or
evaluating, not here. ``--M`` is the total number of domains; the trainers hold out the
last ``--n_test`` (default 1000) of them.

Examples (reproduce the paper's grid: ER{1,2,4} x d{5,10,20,50}, 100000 train + 1000 test):

  python generate_data.py --level hetero       --graph_type ER --s0 1 --d 5  --M 101000
  python generate_data.py --level orderconsist --graph_type ER --s0 2 --d 20 --M 101000
  python generate_data.py --level general      --graph_type ER --s0 4 --d 50 --M 101000
  python generate_data.py --level general --d 5 --M 101000 --noise exp      # non-Gaussian noise

ER4 with d = 5 is impossible (20 edges > the 10 a 5-node DAG can hold) and is rejected.
"""
import argparse

from adag.data import LEVELS, NOISE_TYPES, generate_shard


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", required=True, choices=LEVELS, help="data type")
    p.add_argument("--graph_type", default="ER", choices=["ER", "SF"], help="random graph family")
    p.add_argument("--s0", type=int, default=1, help="expected edges per node (s0 * d edges)")
    p.add_argument("--d", type=int, default=5, help="number of variables")
    p.add_argument("--M", type=int, default=101000, help="total domains (train + test)")
    p.add_argument("--seed", type=int, default=42, help="seed for graphs and weights")
    p.add_argument("--noise", default="gauss", choices=NOISE_TYPES, help="additive noise of the linear SEM")
    p.add_argument("--noise_scale", type=float, default=1.0, help="noise scale (std for gauss)")
    p.add_argument("--w_min", type=float, default=0.5, help="edge weights ~ U(-w_max,-w_min) u U(w_min,w_max)")
    p.add_argument("--w_max", type=float, default=2.0)
    p.add_argument("--root", default="data", help="output root; the shard goes to root/level/...")
    a = p.parse_args()
    generate_shard(a.level, a.graph_type, a.s0, a.d, a.M, a.seed, noise=a.noise,
                   noise_scale=a.noise_scale, w_ranges=((-a.w_max, -a.w_min), (a.w_min, a.w_max)),
                   root=a.root)


if __name__ == "__main__":
    main()
