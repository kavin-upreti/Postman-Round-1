"""
1.5  Benchmark: wall-clock time and peak memory of one attention forward pass,
dense vs sparse, across sequence lengths 512..8192. Forward pass only.

Includes warmup iterations before timing to remove the CUDA cold-start artifact.
Reports the hardware and (if matplotlib is available) saves plots.

Expected result — and the honest finding of this task: masking gives NO speedup
and NO memory saving. The full T x T score matrix is materialized regardless of
pattern (the mask only overwrites cells with -inf after they are computed), so
all three patterns use identical memory and near-identical time. Realizing the
theoretical sparsity would require a custom kernel that never computes the
skipped blocks (out of scope here; that is Task 4's domain).

Run:  python benchmark.py         (needs a CUDA GPU for the memory numbers)
"""

import time
import torch
import torch.nn.functional as F
from masking import build_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
LENGTHS = [512, 1024, 2048, 4096, 8192]
CFGS = {
    "dense":   {"mask": "causal"},
    "sliding": {"mask": "sliding", "w": 64},
    "bigbird": {"mask": "bigbird", "w": 64, "g": 16, "r": 0},
}


def bench_once(size, cfg, warmup=5, iters=20):
    hs = 64
    q = torch.randn(1, size, hs, device=device)
    k = torch.randn(1, size, hs, device=device)
    v = torch.randn(1, size, hs, device=device)
    forbidden = build_mask(cfg["mask"], size, w=cfg.get("w"), g=cfg.get("g"),
                           r=cfg.get("r", 0), device=device)

    def one():
        wei = q @ k.transpose(-2, -1) * hs ** -0.5
        wei = wei.masked_fill(forbidden, float('-inf'))
        d = torch.isinf(wei).all(dim=-1, keepdim=True)
        wei = wei.masked_fill(d, 0.0)
        return F.softmax(wei, dim=-1) @ v

    for _ in range(warmup):
        one()
    if device == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        out = one()
    if device == 'cuda':
        torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == 'cuda' else float('nan')
    del q, k, v, forbidden, out
    if device == 'cuda':
        torch.cuda.empty_cache()
    return ms, peak


if __name__ == "__main__":
    hw = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"hardware: {hw}")
    print(f"{'T':>6} {'mask':>10} {'ms':>8} {'peak_MB':>10}")
    results = {nm: [] for nm in CFGS}
    for T in LENGTHS:
        for nm, cfg in CFGS.items():
            try:
                ms, pk = bench_once(T, cfg)
                print(f"{T:6d} {nm:>10} {ms:8.2f} {pk:10.1f}")
                results[nm].append((T, ms, pk))
            except RuntimeError as e:
                print(f"{T:6d} {nm:>10}  OOM/err: {str(e)[:40]}")
        if device == 'cuda':
            torch.cuda.empty_cache()

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for nm, rows in results.items():
            if not rows:
                continue
            Ts = [r[0] for r in rows]; ms = [r[1] for r in rows]; pk = [r[2] for r in rows]
            a1.plot(Ts, ms, marker='o', label=nm)
            a2.plot(Ts, pk, marker='o', label=nm)
        a1.set_title('Forward-pass time vs T'); a1.set_xlabel('T'); a1.set_ylabel('ms')
        a1.legend(); a1.grid(True, alpha=0.3)
        a2.set_title('Peak memory vs T'); a2.set_xlabel('T'); a2.set_ylabel('peak MB')
        a2.legend(); a2.grid(True, alpha=0.3)
        fig.suptitle(f'Attention cost scaling ({hw})'); fig.tight_layout()
        fig.savefig('results/benchmark.png', dpi=120, bbox_inches='tight')
        print("saved results/benchmark.png")
    except Exception as e:
        print("plotting skipped:", e)
