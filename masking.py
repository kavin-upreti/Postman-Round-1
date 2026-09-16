"""
Mask builders for sparse attention.

Every function returns a (size, size) boolean tensor in the FORBIDDEN
convention: True means the (query, key) pair is blocked and will be set to
-inf before softmax. All three patterns are causal (a token never attends to
the future).

    causal   - full lower triangle (the dense reference mask)
    sliding  - local window of width w
    bigbird  - local window + global tokens + block-random spans

The random component of bigbird is drawn with torch's generator, so
torch.manual_seed(...) makes it reproducible. Masks are built once (see model.py,
where the result is stored as a registered buffer) so the random links are frozen
for the whole run rather than redrawn every forward pass.
"""

import torch


def causal(size, device='cpu'):
    i = torch.arange(size, device=device).unsqueeze(1)
    j = torch.arange(size, device=device).unsqueeze(0)
    return j > i


def sliding(size, w, device='cpu'):
    # forbidden if in the future (j > i) or too far back (j <= i - w)
    i = torch.arange(size, device=device).unsqueeze(1)
    j = torch.arange(size, device=device).unsqueeze(0)
    return (j > i) | (j < i - w + 1)


def bigbird(size, w, g, r, span=32, device='cpu'):
    """local window (w) + global tokens (first g) + r random spans of width `span`.

    Built in the ALLOWED convention, then flipped to forbidden at the end.
    Random spans are drawn from the gap between the global region and the local
    window, so a random link never overlaps a cell that is already allowed.
    """
    allowed = ~sliding(size, w, device)     # local window
    allowed[:, :g] = True                   # global columns: everyone attends to first g
    allowed[:g, :] = True                   # global rows: first g attend to everyone

    if r > 0:
        for i in range(size):
            lo = g
            hi = i - w + 1                  # exclusive top of the gap [lo, hi)
            if hi - lo <= 0:
                continue                    # gap not open yet for early rows
            for _ in range(r):
                max_start = hi - span
                if max_start < lo:
                    start, end = lo, hi     # gap smaller than a span: open what fits
                else:
                    start = torch.randint(lo, max_start + 1, (1,), device=device).item()
                    end = start + span
                allowed[i, start:end] = True

    # re-enforce causal (global rows could have opened future cells above the diagonal)
    i = torch.arange(size, device=device).unsqueeze(1)
    j = torch.arange(size, device=device).unsqueeze(0)
    allowed = allowed & (j <= i)
    return ~allowed


def build_mask(mask_type, size, w=None, g=None, r=0, span=32, device='cpu'):
    if mask_type == "causal":
        return causal(size, device)
    if mask_type == "sliding":
        return sliding(size, w, device)
    if mask_type == "bigbird":
        return bigbird(size, w, g, r, span, device)
    raise ValueError(f"unknown mask_type: {mask_type}")
