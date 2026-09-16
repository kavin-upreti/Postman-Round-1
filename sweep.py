"""
The multi-config sweep that produced the comparison results (1.6), plus the
correctness harness (1.3), NaN demo (1.4), and benchmark (1.5) up front.

This mirrors what was actually run on Colab. It trains every config in CONFIGS
with identical initialization (seed reset before each build, so the mask is the
only difference) and appends each result to a log file as it finishes.

Set SEED to reproduce a specific run (1337 = run 1, 1338 = confirmation run).

Usage:  python sweep.py
"""

import time
import traceback
import torch
from model import GPTLanguageModel
from data import vocab_size, get_batch, decode

BATCH_SIZE, BLOCK_SIZE = 64, 384
MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS = 4000, 500, 20
LR, N_EMBD, N_HEAD, N_LAYER, DROPOUT = 3e-4, 256, 4, 4, 0.15
SEED = 1337
LOG = "results/sweep_log.txt"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

CONFIGS = [
    {"name": "causal",             "mask": "causal"},
    {"name": "sliding_w32",        "mask": "sliding", "w": 32},
    {"name": "sliding_w64",        "mask": "sliding", "w": 64},
    {"name": "bigbird_w32_g16_r0", "mask": "bigbird", "w": 32, "g": 16, "r": 0},
    {"name": "bigbird_w64_g16_r0", "mask": "bigbird", "w": 64, "g": 16, "r": 0},
    {"name": "bigbird_w64_g1_r0",  "mask": "bigbird", "w": 64, "g": 1,  "r": 0},
    {"name": "bigbird_w64_g32_r0", "mask": "bigbird", "w": 64, "g": 32, "r": 0},
    {"name": "bigbird_w32_g16_r2", "mask": "bigbird", "w": 32, "g": 16, "r": 2},
    {"name": "bigbird_w64_g16_r2", "mask": "bigbird", "w": 64, "g": 16, "r": 2},
]


def log(t):
    print(t)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(t + "\n"); f.flush()


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        L = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, BATCH_SIZE, BLOCK_SIZE, device)
            _, loss = model(X, Y)
            L[k] = loss.item()
        out[split] = L.mean().item()
    model.train()
    return out


def run():
    log("=" * 70)
    log(f"SWEEP  block_size={BLOCK_SIZE} max_iters={MAX_ITERS} seed={SEED} device={device}")
    log("=" * 70)
    for cfg in CONFIGS:
        t0 = time.time()
        try:
            torch.manual_seed(SEED)                       # identical init for every config
            model = GPTLanguageModel(vocab_size, N_EMBD, N_HEAD, N_LAYER,
                                     BLOCK_SIZE, DROPOUT, cfg, device).to(device)
            npar = sum(p.numel() for p in model.parameters()) / 1e6
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            for it in range(MAX_ITERS):
                if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
                    L = estimate_loss(model)
                    print(f"[{cfg['name']}] step {it}: train {L['train']:.4f} val {L['val']:.4f}")
                xb, yb = get_batch('train', BATCH_SIZE, BLOCK_SIZE, device)
                _, loss = model(xb, yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            final = estimate_loss(model)
            ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = decode(model.generate(ctx, 500)[0].tolist())
            mins = (time.time() - t0) / 60
            log("-" * 70)
            log(f"CONFIG {cfg['name']}  ({npar:.2f}M, {mins:.1f} min)  "
                f"mask={cfg['mask']} w={cfg.get('w')} g={cfg.get('g')} r={cfg.get('r', 0)}")
            log(f"  FINAL train {final['train']:.4f}  val {final['val']:.4f}")
            log("  sample:"); log(sample); log("-" * 70)
        except Exception:
            log("!" * 70); log(f"CONFIG {cfg['name']} FAILED:"); log(traceback.format_exc()); log("!" * 70)
        try:
            del model, opt
        except Exception:
            pass
        if device == 'cuda':
            torch.cuda.empty_cache()
    log("=" * 70); log("SWEEP DONE"); log("=" * 70)


if __name__ == "__main__":
    run()
