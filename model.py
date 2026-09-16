"""
Character-level GPT with a pluggable attention mask.

This is the base model from Karpathy's "Let's Build GPT", restructured so the
attention mask is chosen per-config and threaded down to each head. The only
line that differs between dense and any sparse pattern is where the mask is
applied inside Head.forward; everything else is identical, which is what keeps
the dense-vs-sparse comparison clean.

The mask is built once per head (in __init__) and stored as a registered buffer,
so it moves to the GPU with the model, stays out of the optimizer, and — for
bigbird — has its random links drawn a single time rather than every step.

Two fixes over the original walkthrough code:
  - attention is scaled by head_size**-0.5 (not n_embd**-0.5); the two coincide
    only in the single-head case, so the bug was invisible there.
  - the transformer block honors the configured n_head instead of a hardcoded value.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from masking import build_mask


class Head(nn.Module):
    """One head of causal self-attention with an arbitrary (frozen) mask."""

    def __init__(self, head_size, n_embd, block_size, dropout, cfg):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        mask = build_mask(cfg["mask"], block_size,
                          w=cfg.get("w"), g=cfg.get("g"),
                          r=cfg.get("r", 0), device='cpu')
        self.register_buffer('mask', mask)   # (block_size, block_size), True = forbidden
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T)
        wei = wei.masked_fill(self.mask[:T, :T], float('-inf'))
        # dead-row guard: a fully-masked row would make softmax return NaN.
        # cannot happen with per-cell masks (the diagonal is always kept) but is
        # kept as insurance and to make the block-skipping case safe. See 1.4.
        dead = torch.isinf(wei).all(dim=-1, keepdim=True)
        wei = wei.masked_fill(dead, 0.0)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout, cfg):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout, cfg) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, cfg):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout, cfg)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, cfg, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, cfg) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok + pos
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
