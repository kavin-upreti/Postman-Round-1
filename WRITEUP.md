# Results and analysis

All experiments use a character-level GPT on TinyShakespeare: n_embd 256, 4 heads,
4 layers, dropout 0.15, batch 64, AdamW 3e-4, 4000 iterations, block_size 384.
Every configuration differs from every other **only** in the attention mask; the
seed is reset before each build so all configs share initialization. Hardware:
a single Colab Tesla T4.

## 2.1 Correctness (1.3) and the NaN edge case (1.4)

The harness compares each sparse path against a slow reference applying the same
mask, and checks two invariants. All patterns pass:

    [PASS] causal            max|diff|=0.00e+00  future_leak=False  dead_row=False
    [PASS] sliding_w8        max|diff|=0.00e+00  future_leak=False  dead_row=False
    [PASS] bigbird_w8g4r2    max|diff|=0.00e+00  future_leak=False  dead_row=False

max|diff| is exactly zero (implemented path == reference), no future cell is ever
allowed, and no row is ever fully masked. That last invariant is why the NaN
(1.4) cannot occur with per-cell masking — every row keeps its diagonal — so it
is demonstrated deliberately in `nan_demo.py`: a forced dead row yields NaN after
softmax, and the dead-row guard (zero all-−∞ rows before softmax) removes it.

## 2.2 Benchmark (1.5): masking gives no speedup and no memory saving

Warmup-corrected, one attention forward pass:

    T        dense           sliding         bigbird
             ms    MB         ms    MB         ms    MB
    512     0.28  11.7       0.21  11.7       0.19  11.7
    1024    0.47  20.1       0.48  20.1       0.48  20.1
    2048    1.70  52.6       1.73  52.6       1.73  52.6
    4096    6.10 180.5       5.63 180.5       5.63 180.5
    8192   22.43 688.0      22.36 688.0      22.40 688.0

**Peak memory is identical across all three patterns at every length**; time is
identical to within noise. `masked_fill` overwrites forbidden cells *after* the
full T×T matrix is computed and stored, so it changes which cells are used, not
which are computed — no FLOPs, no bytes saved. Memory grows ~4× per doubling of T
(the O(T²) curve; a straight line on the log-log plot). Realizing the theoretical
sparsity would need a kernel that never materializes the skipped blocks. The value
of this benchmark is showing precisely *why* a pure-PyTorch mask cannot deliver
the saving the pattern implies. (The first run's 39 ms spike at the smallest dense
point was a one-time CUDA cold-start artifact, fixed here with warmup iterations.)

## 2.3 What was varied, and why

The quality comparison was not a blind grid. Each of the three mask parameters was
chosen and swept to answer a specific question, around a `w=64, g=16, r=0`
baseline, changing one thing at a time so each effect is isolated:

**Window w (swept 32 vs 64).** w controls how far a token sees directly, and with
4 layers the effective reach is roughly w × 4 — so w=32 reaches ~128 characters
and w=64 reaches ~256, a quarter and a half of the 384 context. The question:
does a bigger window help, given that Shakespeare is mostly local? *Expectation:*
either bigger is mildly better (more context) or — if the task is local enough —
bigger mostly adds capacity to overfit and a smaller window acts as a regularizer.
We deliberately kept w below the point where the window covers the whole context
(w=128+ at 4 layers would reach past 384 and make "sparse" meaningless), so the
comparison stays real.

**Global g (swept 1 vs 16 vs 32).** g is the number of shared "hub" tokens every
position can read from and write to. The key thing understood beforehand: g buys
*routing bandwidth*, not reach — one hub is a single fixed-size vector that all
long-range traffic must funnel through (and saturates via superposition), while
more hubs give more independent channels. The question: does more global bandwidth
help? *Expectation, tempered by this setup:* probably little, because our model is
prompt-less and trained on random chunks, so the "first g positions" are
arbitrary characters, not meaningful anchors — they are positional routing hubs
with no privileged content. We swept g across a wide range (1 → 32, i.e. 0.3% →
8% of the context spent on hubs) precisely to see where, if anywhere, added
bandwidth stops mattering.

**Random r (swept 0 vs 2 block-spans).** r adds a few long-range links per row.
The design decision here was to make each link a contiguous 32-column *span*, not
a single cell, because a single random character carries no useful signal — a
random *phrase* might. The question: do random long-range spans add anything on
top of window + global? *Expectation:* small at best; the literature drops random
attention because global tokens usually do the real work and scattered random
access is what hardware handles worst. We tested r at *both* windows (32 and 64)
to see whether random helps more when the window is small (starved of reach) than
when it is large.

The reason for pairing rather than a full cross-product: a full w×g×r grid is many
runs, most testing interactions nobody asked about. Paired comparisons around one
baseline answer the three questions above at a fraction of the cost.

## 2.4 Quality results (1.6) + Expectations

**Expected:**
Window (w) — expected a small effect, maybe favoring the smaller window.
The window controls how many characters back a token can look directly. Our model has 4 layers, and information travels about one window's width per layer, so w=32 effectively reaches ~128 characters back and w=64 reaches ~256. The question was whether reaching further actually helps. Since Shakespeare is mostly short-range — to predict the next character you usually just need the current word and a bit before it — we expected the extra reach of w=64 to add very little real information. Worse, a bigger window gives the model more room to latch onto coincidental long-range patterns in the training text that don't generalize, i.e. to overfit. A smaller window, by forbidding most connections, quietly acts like a regularizer. So the prediction was: w=32 and w=64 land close together, and there's a real chance the smaller one comes out slightly ahead — the opposite of the naive "more context is always better." That's roughly what we saw.

Global (g) — expected little to no benefit, despite the theory saying otherwise.
Global tokens are shared "hub" positions that every token can read from and write to, giving any two distant tokens a two-hop shortcut. In principle that's powerful and cheap. But two things made us expect it to fall flat here. First, this task barely has long-range dependencies to route through the hub — the useful information is all nearby, so the shortcut solves a problem the task doesn't really have. Second, and more specific to our setup: because we train on random chunks with no prompt, the "first g positions" that serve as global tokens are just arbitrary characters, not meaningful anchors like a title or a speaker name. They're routing slots with no special content. So we predicted g=1, 16, and 32 would come out nearly identical, and if anything a larger g might hurt slightly, because spending 8% of every token's attention (at g=32) on fixed positions is a mild waste. The result matched: a tiny monotonic "more global is worse" trend, but well within noise.

Random (r) — expected the weakest and least trustworthy effect of the three.
Random links add a few long-range connections per token, chosen arbitrarily. We already knew from reasoning (and the literature) that a single random link is nearly useless — which is why we made each one a 32-character span instead of one cell, so it at least carries a phrase's worth of information. Even so, we expected random to barely move the loss, because its value is only statistical (many random links together slightly shrink the average distance across the sequence), and on a short-range task that barely matters. We also guessed it might help a little more at the small window (w=32), where the model is more starved for reach, than at the large window. Our honest prediction was: negligible effect, and whatever we see is probably noise. This turned out exactly right — and importantly, the first run looked like it found a real pattern (random helped at w=32), but the second seed flipped the sign, confirming it was luck.

The idea underneath all three: every effect should be small, because the task is short-range and four layers of even a modest window already capture what's needed. Global and random specifically should underperform their textbook promise — global because there's nothing long-range to route and its hubs here are content-free, random because individual links carry almost nothing. The window was the only place we expected a visible (if still modest) signal, and it was.


**Actual:**
Run 1 (seed 1337), final validation loss, sorted:

    bigbird w32 g16 r2   1.5098
    sliding w32          1.5129
    bigbird w32 g16 r0   1.5188
    sliding w64          1.5299
    bigbird w64 g1  r0   1.5318
    bigbird w64 g16 r0   1.5364
    bigbird w64 g32 r0   1.5387
    bigbird w64 g16 r2   1.5390
    causal (dense)       1.5458

Taken alone, this table invites four confident stories: sparse beats dense,
smaller window wins, more global hurts, random helps at w32. The confirmation run
(seed 1338) shows how many survive.

**The noise floor — the number that governs everything.** The *same* causal
config trained at two seeds gave 1.5458 (seed 1337) and 1.5304 (seed 1338): a
**0.015** gap from the random seed alone. So run-to-run noise here is ~0.015 nats,
and any effect smaller than that cannot be told apart from luck. For scale, these
are cross-entropy nats/char (ln 65 ≈ 4.17 is random guessing); 0.015 nats is a
perplexity change of ~0.05, invisible in the generated text — which is exactly why
samples from every config read alike (caps names, verse structure, the same rate
of malformed words). Establishing this floor is the single most important step in
the analysis; without it every small gap above would have been over-read.

**Window (w32 vs w64): a real but small effect.**

    bigbird r0:  seed1337  w32 1.5188 vs w64 1.5364   (gap 0.018)
                 seed1338  w32 1.5051 vs w64 1.5156   (gap 0.011)
    sliding:     seed1337  w32 1.5129 vs w64 1.5299   (gap 0.017)

w32 was ahead in *every* pair, across both seeds. The individual gaps straddle
the 0.015 floor, so no single one is decisive — but the *consistency* across
several independent comparisons all pointing the same way is much harder to
explain by luck than one gap would be. **Verdict: suggestive, leaning real.** This
matches the expectation that Shakespeare is local enough that w=64's extra reach
mostly adds overfitting capacity rather than signal, so the smaller window
regularizes slightly better. It is the opposite of the naive "more context is
better," and it is the clearest signal in the experiment.

**Global (g1 vs g16 vs g32): no benefit, as expected.** Run 1: g1 1.5318 < g16
1.5364 < g32 1.5387 — monotonic, and *more* global is slightly *worse*. But the
whole spread (g1→g32) is 0.007, well under the 0.015 floor, so it was not
re-tested. **Verdict: no benefit observed, possibly a tiny penalty, within
noise.** This is exactly what the setup predicted: with content-free positional
hubs on a mostly-local task, there is little for global routing to carry that the
window and depth do not already provide, and spending 8% of every row's attention
(g=32) on fixed positions is if anything a small waste. A negative result, but a
*predicted* one — the disproportionate benefit global tokens have in principle
requires long-range dependencies this task does not contain.

**Random (r0 vs r2): the effect that dissolved.** Run 1 suggested an interesting
interaction — r2 helped at w32 (−0.009) and was neutral at w64. The confirmation
run destroyed it:

    w32:  seed1337  r0→r2  = -0.009 (r2 better)
          seed1338  r0→r2  = +0.027 (r2 worse)
    w64:  seed1337  r0→r2  = +0.003 (r2 worse)
          seed1338  r0→r2  = -0.003 (r2 better)

The sign of the effect **flips between seeds at both windows** — the signature of
noise, not signal. **Verdict: block-random has no reliable effect on this task,
and its apparent sign is unstable.** This is precisely why the confirmation run
was worth doing: the seed-1337 "interaction" looked like a genuine, publishable
finding, and it was luck. It also matches the literature's decision to drop random
attention. Had we reported run 1 alone, we would have claimed an effect that does
not exist.

## 2.5 Conclusion

**On this task, no attention-pattern effect reliably exceeds the ~0.015 noise
floor.** Every configuration — dense, sliding, BigBird, across window sizes,
global counts, and random settings — lands between 1.50 and 1.55. The only
directionally consistent signal is mild: sparse tended to edge out dense, and the
smaller window tended to edge out the larger, both suggestive rather than proven.
Global tokens showed no benefit (as predicted), and random showed no stable effect
at all (its sign flipped across seeds).

This is the correct result, not a disappointing one, and it is what the task
framing anticipated: a short-range benchmark where the attention pattern barely
matters, because almost all predictive signal in character-level Shakespeare lives
within a few dozen characters and four layers of a modest window already capture
it. The regime where sparse attention is designed to pay off — long contexts with
genuine long-range dependencies — is exactly what this benchmark does not
exercise. The honest contribution of the experiment is therefore methodological:
establishing a noise floor and refusing to claim effects that do not clear it,
which turned a plausible-looking random-attention "finding" back into noise.

**What each pattern loses.** Sliding loses every dependency beyond w × n_layer;
here that is nearly free. BigBird restores a long-range path but a narrow one —
everything distant routes through a handful of fixed-capacity global vectors (a
superposition bottleneck) or a few random links, so it carries *some* long-range
information, not much at once.

**Why global tokens matter disproportionately (in principle).** A negligible
number of them collapses the path length between any two positions from many
layers to two hops (write to the hub, read from it). The cost is tiny, the
connectivity gain large. On *this* task that connectivity is not needed, so the
benefit does not materialize — an honest and instructive negative.

**Was sparse worse?** No — if anything marginally better than dense (dense had the
highest loss), most plausibly because the sparse masks act as a mild regularizer
on a task that does not need long range. But the margins are at or below the noise
floor, so the safe statement is that **sparse matched dense at no measurable
quality cost**, while — in a real kernel rather than this masking implementation —
it would offer large efficiency gains at long context.

## 2.6 Limitations

- The sparse implementations are correct but **not faster and not lighter** —
  per-cell masking computes the full matrix; real savings need a kernel.
- A single extra seed gives a noise *estimate*, not a rigorous variance; effects
  near the floor (window, global) would need several seeds to settle definitively.
  The choice not to run them was deliberate given their small size, and they are
  reported as suggestive rather than proven.
- Global tokens here are positional routing hubs with arbitrary content (no
  prompt, random training chunks), which limits their benefit relative to the
  original BigBird encoder setting and is part of why their measured effect was nil.