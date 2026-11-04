"""
Compare ADAG with per-domain baselines on the held-out domains of a dataset.

Every method sees exactly the same observations (same shard, same noise seed, same n)
and is scored with the same metrics at the same threshold (|w| >= 0.3), so the rows of
the printed table are directly comparable.

  adag      zero-shot: one forward pass of a trained checkpoint (--ckpt)
  notears   NOTEARS, linear, L2 loss, lambda1 = --lambda1   (pip install git+https://github.com/xunzheng/notears)
  dagma     DAGMA, linear, L2 loss, lambda1 = --lambda1     (pip install dagma)
  avici     AVICI, pretrained 'scm-v0' (edge probabilities, cut at 0.5; no weights)  (pip install avici)

  python run_baselines.py --methods adag notears dagma --ckpt demo/checkpoints/adag_d5_n1000.ckpt \
      --data_root demo/data --level hetero --s0 1 --d 5 --n 1000

Baselines run on CPU with one process per core (--workers); ADAG runs on the GPU if there
is one. Per-domain results go to --out_dir/<method>/, and the table is printed and saved as
--out_dir/comparison_<tag>.md.  Use --limit to try a few domains first.
"""
import argparse, contextlib, io, json, os, time

import numpy as np

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from adag.training import add_data_args, add_model_args
from adag.utils import METRIC_KEYS, graph_metrics, summarize

WEIGHTED = {"adag": True, "notears": True, "dagma": True, "avici": False}
THRESHOLD = {"adag": None, "notears": None, "dagma": None, "avici": 0.5}   # None -> --w_threshold


# ----------------------------------------------------------------------------- baselines (one domain)
_G = {}
def _init(method, path, n, noise_seed, lambda1):
    from adag.data import Shard
    _G.update(method=method, shard=Shard(path, n=n, device="cpu", noise_seed=noise_seed), lambda1=lambda1)
    if method == "avici":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import avici
        with contextlib.redirect_stdout(io.StringIO()):
            _G["avici"] = avici.load_pretrained(download="scm-v0", verbose=False)


def _fit(method, X, lambda1):
    if method == "notears":
        import notears.linear as nl
        if not getattr(nl, "_adag_patched", False):
            # Some local copies of notears pass a callback that breaks on scipy >= 1.11 and
            # return (W, trajectory); upstream does neither. Strip the callback, keep W.
            import scipy.optimize as so, types
            nl.sopt = types.SimpleNamespace(minimize=lambda *a, **k: so.minimize(
                *a, **{kk: v for kk, v in k.items() if kk != "callback"}))
            nl._adag_patched = True
        out = nl.notears_linear(X, lambda1=lambda1, loss_type="l2", w_threshold=0.0)
        return out[0] if isinstance(out, tuple) else out
    if method == "dagma":
        from dagma.linear import DagmaLinear
        with contextlib.redirect_stderr(io.StringIO()):
            return DagmaLinear(loss_type="l2").fit(X, lambda1=lambda1, w_threshold=0.0)
    if method == "avici":
        return np.asarray(_G["avici"](x=X))
    raise ValueError(method)


def _one(i):
    sh = _G["shard"]
    X, _ = sh.batch([i], n=sh.n)
    X = X[0].numpy().astype(np.float64)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        W = _fit(_G["method"], X, _G["lambda1"])
    return int(i), np.asarray(W, dtype=np.float32), time.time() - t0


def run_baseline(method, path, test, n, noise_seed, lambda1, workers, log_every=100):
    import multiprocessing as mp
    W_all, rt = np.zeros((len(test), 0, 0), np.float32), np.zeros(len(test))
    pos = {int(i): k for k, i in enumerate(test)}
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(workers, len(test)), initializer=_init,
                                      initargs=(method, path, n, noise_seed, lambda1)) as pool:
        for k, (i, W, dt) in enumerate(pool.imap_unordered(_one, test.tolist(), chunksize=1)):
            if W_all.shape[1] == 0:
                W_all = np.zeros((len(test),) + W.shape, np.float32)
            W_all[pos[i]] = W; rt[pos[i]] = dt
            if (k + 1) % log_every == 0 or k + 1 == len(test):
                print(f"   {method}: {k + 1}/{len(test)} domains, {(time.time() - t0) / 60:.1f} min", flush=True)
    return W_all, rt


# ----------------------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["adag", "notears", "dagma"],
                   choices=list(WEIGHTED), help="methods to run, in table order")
    p.add_argument("--ckpt", default=None, help="ADAG checkpoint (required for method adag)")
    add_data_args(p); add_model_args(p)
    p.add_argument("--lambda1", type=float, default=0.02, help="L1 weight for NOTEARS / DAGMA")
    p.add_argument("--w_threshold", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None, help="only the first K test domains")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out_dir", default="results")
    a = p.parse_args()

    from adag.data import Shard, shard_path
    path = shard_path(a.data_root, a.level, a.graph_type, a.s0, a.d, a.M, a.data_seed)
    shard = Shard(path, n=a.n, device="cpu", noise_seed=a.noise_seed)
    _, test = shard.split(a.n_test)
    if a.limit:
        test = test[:a.limit]
    W_true = shard.batch(test, n=2)[1].numpy().astype(np.float64)
    tag = f"{a.level}_{a.graph_type}{a.s0}_d{a.d}_n{a.n}_test{len(test)}"
    print(f">> {shard}\n>> {len(test)} test domains, threshold {a.w_threshold}, methods {a.methods}")

    table = []
    for method in a.methods:
        print(f"\n== {method}")
        if method == "adag":
            import torch
            from adag import ADAG
            from evaluate_adag import evaluate
            if not a.ckpt:
                raise SystemExit("--ckpt is required for method adag")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gshard = Shard(path, n=a.n, device=device, noise_seed=a.noise_seed)
            model = ADAG.from_checkpoint(a.ckpt, a.n_heads, a.dk, a.n_layers, a.d, a.n, device=device)
            rows, W_est, wall = evaluate(model, gshard, test, a.n, 100, a.w_threshold)
            rt = np.full(len(test), wall / len(test))
        else:
            W_est, rt = run_baseline(method, path, test, a.n, a.noise_seed, a.lambda1, a.workers)
            thr = THRESHOLD[method] or a.w_threshold
            rows = [graph_metrics(W_est[k], W_true[k], thr) for k in range(len(test))]
            if not WEIGHTED[method]:
                for r in rows:
                    r["rel_error"] = float("nan")
        summ = summarize(rows)
        summ.update(method=method, runtime_s=dict(mean=float(rt.mean()), std=float(rt.std())),
                    domains=len(test), threshold=THRESHOLD[method] or a.w_threshold)
        out = os.path.join(a.out_dir, method); os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, f"A_est_{tag}.npy"), W_est)
        np.savetxt(os.path.join(out, f"metrics_{tag}.csv"), np.array([[r[k] for k in METRIC_KEYS] for r in rows]),
                   delimiter=",", header=",".join(METRIC_KEYS), comments="")
        with open(os.path.join(out, f"summary_{tag}.json"), "w") as f:
            json.dump(summ, f, indent=2)
        table.append(summ)
        print(f"   SHD {summ['shd']['mean']:.3f}  F1 {summ['f1']['mean']:.3f}  "
              f"rel.err {summ['rel_error']['mean']:.3f}  {1e3 * rt.mean():.2f} ms/domain")

    ms = lambda s, p=3: f"{s['mean']:.{p}f} ± {s['std']:.{p}f}"
    lines = [f"**{a.level}, {a.graph_type}{a.s0}, d = {a.d}, n = {a.n}** ({len(test)} test domains, "
             f"{int((W_true[0] != 0).sum())} true edges per domain, threshold {a.w_threshold})", "",
             "| Method | SHD | Edges | F1 | Rel. error | Time / domain |", "|---|---|---|---|---|---|"]
    for s in table:
        rel = "n/a" if np.isnan(s["rel_error"]["mean"]) else ms(s["rel_error"])
        lines.append(f"| {s['method'].upper() if s['method'] != 'adag' else 'ADAG'} | {ms(s['shd'])} | "
                     f"{ms(s['nnz'], 2)} | {ms(s['f1'])} | {rel} | {1e3 * s['runtime_s']['mean']:.2f} ms |")
    md = "\n".join(lines)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"comparison_{tag}.md"), "w") as f:
        f.write(md + "\n")
    print("\n" + md)
    print(f"\n>> table saved to {a.out_dir}/comparison_{tag}.md")


if __name__ == "__main__":
    main()
