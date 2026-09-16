# Note on commit history

**Disclosure:** development was done iteratively in Google Colab and a local
editor, and the git history was assembled in logical stages near the end rather
than committed continuously as the work progressed. The stages below reflect the
real order in which the work was actually built. This note is included in the
interest of honesty about process — the same standard applied to the results
(effects within noise are reported as such rather than overclaimed).
Sorry!

## The order the work was actually developed

1. **Dense baseline.** Started from Karpathy's *Let's Build GPT* character-level
   model. 

2. **Pluggable mask.** Refactored the attention head so the mask is chosen from
   outside and threaded down, instead of a hardcoded causal triangle. Verified
   the refactor trained identically to the original.

3. **Sliding window.** Added the sliding-window mask and confirmed it by eye on a
   small grid, then trained it. Swept window width (w=2 vs w=8) to locate where a
   too-small window degrades quality.

4. **BigBird mask.** Added local + global + random. First with per-cell random.

5. **Build-once masks.** Moved mask construction into `__init__` and stored it as
   a buffer, so BigBird's random links are drawn a single time and frozen (a mask
   rebuilt every step would redraw random links every step, which the model cannot
   learn against). Switched random draws to torch's generator for reproducibility.

6. **Block-random.** Replaced per-cell random with contiguous 32-column spans
   drawn from the gap between global and window, since a single random character
   carries no useful signal (this is also why real BigBird uses random *blocks*).
   Parameterized w/g/r per config.

7. **Correctness, NaN, benchmark.** Added the correctness harness (1.3), the NaN
   demonstration and fix (1.4), and the time/memory benchmark with plots (1.5).

8. **Sweep.** Added the multi-config driver with identical-init seeding and
   per-config logging; ran the 9-config sweep at seed 1337 (block_size 384).

9. **Confirmation run.** Added a second run at seed 1338 to establish a noise
   floor and test whether the small global/random effects reproduce. They did not
   (random's sign flipped across seeds), which corrected an over-claim from the
   first run.

10. **Documentation.** Wrote the ideation notes, the results writeup, and this
    repository.

## Suggested commit sequence (if reconstructing)

    git init
    # stage the files relevant to each step below, then:
    git commit -m "Dense GPT baseline; fix head_size scaling and n_head bugs"
    git commit -m "Make attention mask pluggable; thread mask_type through model"
    git commit -m "Add sliding-window mask; window-size sweep"
    git commit -m "Add BigBird mask (local + global + random)"
    git commit -m "Build masks once as buffers; freeze random links, torch RNG"
    git commit -m "Block-span random; parameterize w/g/r per config"
    git commit -m "Add correctness harness, NaN demo, benchmark with plots"
    git commit -m "Add sweep driver with identical-init seeding and logging"
    git commit -m "Add confirmation run at second seed (noise floor)"
    git commit -m "Add ideation notes, results writeup, README"
