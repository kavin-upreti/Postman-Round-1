"""
Train a single model with a chosen mask and report final loss + a sample.
This is the minimal, runnable entry point for 1.1 / 1.6 (train one config).

Usage examples:
    python train.py causal
    python train.py sliding --w 32
    python train.py bigbird --w 64 --g 16 --r 2

For the full multi-config comparison that produced the results, see sweep.py.
"""

import argparse
import torch
from model import GPTLanguageModel
from data import vocab_size, get_batch, decode

# ---- fixed hyperparameters (shared across all runs) ----
BATCH_SIZE   = 64
BLOCK_SIZE   = 384
MAX_ITERS    = 4000
EVAL_INTERVAL = 500
EVAL_ITERS   = 20
LR           = 3e-4
N_EMBD       = 256
N_HEAD       = 4
N_LAYER      = 4
DROPOUT      = 0.15
SEED         = 1337

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, BATCH_SIZE, BLOCK_SIZE, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask", choices=["causal", "sliding", "bigbird"])
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--g", type=int, default=16)
    ap.add_argument("--r", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    cfg = {"mask": args.mask, "w": args.w, "g": args.g, "r": args.r}
    torch.manual_seed(args.seed)

    model = GPTLanguageModel(vocab_size, N_EMBD, N_HEAD, N_LAYER,
                             BLOCK_SIZE, DROPOUT, cfg, device).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M params  |  device:", device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            L = estimate_loss(model)
            print(f"step {it}: train {L['train']:.4f}  val {L['val']:.4f}")
        xb, yb = get_batch('train', BATCH_SIZE, BLOCK_SIZE, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(ctx, 500)[0].tolist()))


if __name__ == "__main__":
    main()
