# Sparse Attention — Ideation and Reasoning

The thinking behind the project: what the task asks, how the ideas work, what we
tried and rejected, and why each choice was made. Results live in the writeup;
this is just the reasoning.

## The problem

Attention compares every token to every other token, making a T×T grid of
scores. That grid grows with T² — double the sequence, quadruple the work (about
67 million entries at T=8192, per head, per layer). That is why long context is
expensive. Sparse attention bets that most of those scores are near zero anyway,
because language is mostly local, so it decides in advance which token pairs may
interact and skips the rest. That decision is a **mask** over the grid.

The task is not "make it faster." It is: build sparse attention by hand, prove it
is correct, measure its cost, and report honestly — including if sparse turns out
no better than dense. The grading rewards understanding, not finishing, so the
whole approach was depth and honesty over a forced win.

## Understanding attention first

**Why three vectors (query, key, value).** For each token, attention builds a new
vector summarizing earlier tokens. That needs two decisions: *how much* to take
from each token, and *what* to take. Query and key answer "how much" — a query is
what a token looks for, a key is what it advertises, and their dot product is
relevance. They use different learned projections, so relevance can be one-sided
(a verb can seek its subject without the subject equally seeking the verb) and
learned rather than raw similarity. Value answers "what" — the actual content
handed over, kept separate from the key because being *findable* and being worth
*delivering* are different jobs. Mechanically, `q @ k.T` gives a pure score grid
with no content left in it; softmax makes each row into weights; `wei @ v` puts
content back, since v is the only thing still carrying it.

**Why the 1/√d scaling.** A dot product over d dimensions sums d products, and
summing d things multiplies variance by d. Too much variance makes softmax
saturate and gradients die. Dividing by √d cancels that. It is the same idea as
the constants in weight init — keeping the signal near unit scale, not magic.

**The base model** was Karpathy's small character-level GPT on TinyShakespeare.
Two latent bugs were caught and fixed: the attention scale used the full width
instead of the per-head width (invisible with one head), and a block ignored the
configured head count.

## The sparse patterns

**Sliding window:** a token attends to itself and the w tokens behind it. Linear
cost. What it loses: information travels only ~w positions per layer, so reach is
about w × layers — beyond that there is no path at all.

**Depth vs width.** A wide window with one layer and a narrow window with many
layers can reach the same distance but are not equal. One wide layer has direct
access but only one round of processing. A deep narrow stack reaches the same
distance through several rounds, each building on the last, so it composes
hierarchy: characters → words → phrases → relations. Depth buys composition;
width buys sharp direct access. This is also why sliding loses little here — four
layers of a modest window compose local information well, and Shakespeare barely
has the long-range signal sliding throws away.

**BigBird = local + global + random.** Global tokens are shared hubs every token
can read from and write to; they work both ways (a hub reads the whole sequence,
then every token reads the hub), giving any two distant tokens a two-hop
shortcut. A tiny number of them collapses long distances to two hops — a huge
effect for a tiny cost, which is why they "matter disproportionately." Their
limit: a hub is one fixed-size vector, so it can only hold so much before stored
things blur together. Random links are the weakest ingredient — one link carries
almost nothing, useful only statistically. The literature often drops random
attention for this reason.

**A caveat for our setup:** because the model is prompt-less and trained on
random chunks, the "first g positions" used as global hubs are arbitrary
characters, not meaningful anchors. So they route, but carry no special content —
a reason to expect them to help less here than in the original BigBird.

## Ideas we rejected

- **Learning the sparse pattern from data:** chicken-and-egg — knowing which
  cells matter needs the scores, which is the cost we are avoiding. And averaged
  patterns collapse back to "local plus a few anchors," which BigBird already is.
- **True block-skipping (never computing skipped blocks):** the only thing that
  gives real savings, but pure PyTorch cannot cheaply skip scattered blocks — it
  needs a custom kernel (out of scope). So we accepted up front that our masks
  save no time or memory, and that the benchmark would show exactly that.
- **Per-cell random:** replaced with 32-column *spans*, since a single random
  character carries no signal but a random phrase might (this is also why real
  BigBird uses random blocks).

## The NaN edge case

Softmax computes `exp(s − max)`. A fully-masked row has max = −∞, so the first
step is (−∞) − (−∞) = NaN. But with per-cell masking this cannot happen — every
row keeps its own diagonal — so we demonstrate it deliberately and fix it with a
guard (zero any all-−∞ row before softmax). It only arises for real under
block-skipping, where a coarse block decision can drop a row's diagonal.

## Design decisions

- **Pluggable mask:** the one line that matters is where the mask is applied, so
  the mask is built outside and passed in — a boolean grid, True where forbidden,
  applied the same way for every pattern. So dense and sparse differ only in the
  grid.
- **Build the mask once:** BigBird's random links must be frozen — a mask rebuilt
  each step would redraw them each step, and the model cannot learn a structure
  that keeps changing. Built once and stored as a buffer (which also moves it to
  the GPU and keeps it out of the optimizer). Drawn with torch's generator so the
  seed controls it.
- **Clean experiment:** context lengthened so sparsity actually matters; configs
  built as paired comparisons around one baseline so each isolates one effect;
  the seed reset before every model so all share the same starting weights and
  the mask is the only difference; all other hyperparameters fixed.
- **A second run** at a different seed, added afterward, to measure the noise
  floor and check whether small effects reproduce — because one run cannot tell a
  small real effect from luck.

## What this is and is not

It is an honest, from-scratch implementation: manual dense reference, sliding and
BigBird masks built by hand and understood piece by piece, a correctness test, a
demonstrated-and-fixed NaN case, a truthful benchmark, and a controlled quality
comparison where each conclusion is stated only at the confidence the data
supports. It is not a fast kernel — per-cell masking cannot speed anything up,
and we say so plainly. And it does not claim every small difference is real; a
second seed keeps that honest.