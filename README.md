# Sparse Attention from Scratch

A from-scratch implementation and analysis of sparse attention for a
character-level GPT, built for the Postman AI/ML task (Task 1). It implements
manual dense attention as a reference, two sparsity patterns (sliding window and
BigBird-style local + global + random), a correctness harness, a demonstrated
NaN edge case and fix, a time/memory benchmark, and a controlled quality
comparison across patterns — with results reported at the confidence the data
actually supports.

## TL;DR of findings

- Correctness: all masks pass (exact match to reference, no future leaks, no dead rows).
- Benchmark: masking gives **no speedup and no memory saving** — the full T×T
  matrix is still computed; only its cells are overwritten. Realizing the saving
  needs a custom kernel (out of scope). Memory scales O(T²) as expected.
- Quality: on TinyShakespeare, **no attention-pattern effect reliably exceeds the
  run-to-run noise floor (~0.015 nats)**. Sparse matched dense at no measurable
  quality cost; smaller windows and sparse-over-dense were mildly but not
  conclusively favored; block-random showed no stable effect (its sign flipped
  across seeds). This is the expected regime for a short-range task.

Full reasoning is in `IDEATION.md`; full results and analysis in `WRITEUP.md`.

## Repository layout

    model.py          GPT with a pluggable attention mask (dense reference = 1.1)
    masking.py        causal / sliding / bigbird mask builders (1.2)
    data.py           TinyShakespeare loading, char encoding, batching
    correctness.py    1.3  correctness harness (prints max|diff| + invariants)
    nan_demo.py       1.4  NaN demonstration and fix
    benchmark.py      1.5  wall-clock + peak memory vs sequence length, with plots
    train.py          train a single config (1.1 / 1.6 entry point)
    sweep.py          multi-config comparison that produced the results (1.6)
    WRITEUP.md        the results: measurements and honest analysis (1.7) with the discussion behind the entire task
    COMMIT_HISTORY.md note on how the repo history was assembled
    results/          raw run logs and benchmark plots

## Setup

    pip install torch matplotlib
    # TinyShakespeare (place input.txt in the repo root):
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## How to run

Quick, CPU-friendly checks (seconds):

    python correctness.py     # 1.3  — should print ALL PASS
    python nan_demo.py        # 1.4  — shows NaN, then the guard fixing it

Benchmark (needs a CUDA GPU for the memory numbers; ~1 minute):

    python benchmark.py       # 1.5  — table + results/benchmark.png

Train one config (~20 min on a T4 at the default block_size=384):

    python train.py causal
    python train.py sliding --w 32
    python train.py bigbird --w 64 --g 16 --r 2

Full comparison sweep (what produced the results; ~3.5 h on a T4):

    python sweep.py           # 1.6  — logs to results/sweep_log.txt
    # set SEED at the top to 1337 (run 1) or 1338 (confirmation run)

## Configuration

Hyperparameters are constants at the top of `train.py` / `sweep.py`
(block_size 384, 4 layers, 4 heads, n_embd 256, 4000 iters). Mask parameters:
`w` = window width, `g` = number of global tokens, `r` = number of random spans
(each a contiguous 32-column block). All configs share initialization (seed reset
before each build) so the mask is the only variable.

## Environment

Developed on Apple Silicon (MPS) for iteration; all training and benchmarking
results were produced on a Google Colab Tesla T4 (CUDA). The device is
auto-selected (cuda → mps → cpu).
