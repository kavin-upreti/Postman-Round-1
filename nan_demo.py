"""
1.4  NaN handling: demonstrate, then fix.

With per-cell masking the all-masked-row NaN cannot occur, because every row
keeps its diagonal (a token can always attend to itself). So we construct the
failure deliberately to show (a) that it produces NaN and (b) that the dead-row
guard fixes it.

When it arises for real: block-level skipping, where a coarse per-block decision
can drop the block that happened to contain a row's diagonal, leaving that row
fully masked.

Run:  python nan_demo.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)
T = 8
wei = torch.randn(T, T)

forbidden = torch.zeros(T, T, dtype=torch.bool)
forbidden[3, :] = True   # row 3 fully masked

print("=" * 60)
print("1.4  NaN DEMONSTRATION")
print("=" * 60)

# without the guard: softmax over an all -inf row -> NaN
w = wei.masked_fill(forbidden, float('-inf'))
bad = F.softmax(w, dim=-1)
print("without guard: row 3 contains NaN =", torch.isnan(bad).any().item(), "(expected True)")

# with the guard: zero dead rows before softmax
w = wei.masked_fill(forbidden, float('-inf'))
dead = torch.isinf(w).all(dim=-1, keepdim=True)
w = w.masked_fill(dead, 0.0)
ok = F.softmax(w, dim=-1)
print("with guard:    row 3 contains NaN =", torch.isnan(ok).any().item(), "(expected False)")
print("with guard:    row 3 =", [round(x, 3) for x in ok[3].tolist()], "(uniform, safe)")
print()
print("Cause: softmax computes exp(s - max). An all -inf row has max = -inf, so")
print("the first op is (-inf) - (-inf) = NaN, which poisons the row. The guard")
print("detects the dead row and zeros it before softmax.")
print("=" * 60)
