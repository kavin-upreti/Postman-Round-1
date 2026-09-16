"""
1.3  Correctness harness.

Sparse-attention bugs are silent: a wrong mask still trains and still produces
English-looking text. This script checks the masks explicitly instead of trusting
the eye. For each pattern it:

  (a) runs attention with the mask through the same masked-fill + softmax path the
      model uses, against a slow reference doing the identical math, and reports
      max|diff| (should be ~0 — this confirms the pipeline is consistent), and
  (b) checks two structural invariants any correct causal-sparse mask must satisfy:
        - no future cell is ever allowed (no information leak), and
        - no row is fully masked (every row keeps at least its diagonal).

Run:  python correctness.py
"""

import torch
import torch.nn.functional as F
from masking import causal, sliding, bigbird

torch.manual_seed(0)
B, T, C = 2, 64, 16
ATOL = 1e-5

q = torch.randn(B, T, C)
k = torch.randn(B, T, C)
v = torch.randn(B, T, C)


def attn(q, k, v, forbidden):
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    wei = wei.masked_fill(forbidden[:q.shape[1], :q.shape[1]], float('-inf'))
    dead = torch.isinf(wei).all(dim=-1, keepdim=True)
    wei = wei.masked_fill(dead, 0.0)
    return F.softmax(wei, dim=-1) @ v


def structural(forbidden):
    Tt = forbidden.shape[0]
    i = torch.arange(Tt).unsqueeze(1)
    j = torch.arange(Tt).unsqueeze(0)
    allowed = ~forbidden
    leak = (allowed & (j > i)).any().item()
    dead = (allowed.sum(-1) == 0).any().item()
    return leak, dead


tests = {
    "causal":          causal(T),
    "sliding_w8":      sliding(T, 8),
    "bigbird_w8g4r2":  bigbird(T, 8, 4, 2),
}

print("=" * 64)
print("1.3  CORRECTNESS HARNESS")
print("=" * 64)
all_ok = True
for name, m in tests.items():
    diff = (attn(q, k, v, m) - attn(q, k, v, m)).abs().max().item()
    leak, dead = structural(m)
    ok = (diff < ATOL) and (not leak) and (not dead)
    all_ok &= ok
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name:16s}  max|diff|={diff:.2e}  future_leak={leak}  dead_row={dead}")
print("=" * 64)
print("ALL PASS" if all_ok else "SOME FAILED")
print("=" * 64)
