# ByteMoE feasibility gates (E0–E2)

This repository starts with the smallest useful decision harness for the
ByteMoE design: exact block reconstruction (E0), importance-ordered prefix
viability (E1), and transfer/kernel efficiency (E2). It deliberately does not
yet implement learned scheduling, calibration, cache policies, or a simulator.

## What the scripts test

- `scripts/e0_exactness.py` uses an additive SwiGLU decomposition, replaces one
  routed Hugging Face expert with the sum of all its blocks, and checks direct
  expert error plus end-to-end next-token agreement.
- `scripts/e1_prefix_viability.py` replaces one routed expert by an
  importance-ordered 25%/50% neuron prefix and compares its next-token
  agreement with three random prefixes of the same size. The importance score
  is data-free (projection norms), so a pass only justifies the next phase; it
  is not the final learned importance predictor.
- `scripts/e2_microbenchmark.py` measures pinned-memory H2D copy and SwiGLU
  compute time for candidate widths and a whole-expert reference width.

The Hugging Face adapter assumes each expert is an invoked module containing
`gate_proj`, `up_proj`, and `down_proj` linear layers. List discovered experts
before an experiment. If a model represents all experts in a fused tensor, add
a model-specific adapter rather than trusting these measurements.

## Run on a T4 Colab instance

1. Open `ByteMoE_E0_E2_Colab.ipynb` in Colab with a T4 GPU enabled. The
   notebook mounts Google Drive and persists the Hugging Face cache and
   `results/` under `MyDrive/ByteMoEColab`; it clones the Git repository to the
   local Colab disk on each new runtime.
2. Install PyTorch appropriate to the Colab CUDA runtime, then install the
   remaining packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate with Hugging Face only if the selected model requires it:

   ```bash
   huggingface-cli login
   ```

4. Verify the pure decomposition logic:

   ```bash
   python -m pytest -q
   ```

   The Colab notebook also builds a persistent file of 512 distinct WikiText-2
   prompts for E1. For command-line runs, create an equivalent one-prompt-per-
   line file and provide it with `--prompt-file`.

5. List model experts. This downloads the model on first run; the download is
   cached in Drive and can resume after a Colab disconnect:

   ```bash
   python -m scripts.e0_exactness --list-experts
   ```

6. Run E0. The default expert index `-1` automatically selects the most-active
   expert for the supplied prompts:

   ```bash
   python -m scripts.e0_exactness --expert-index -1 --blocks 16 --prompt-copies 4
   ```

   Continue only if `passed: True`; inspect
   `results/e0_exactness.json` for direct and end-to-end errors.

7. Run E1 using the same expert:

   ```bash
   python -m scripts.e1_prefix_viability --expert-index -1 --fractions 0.25 0.5 --prompt-copies 8
   ```

   Continue only if `passed: True` and an ordered prefix beats the mean random
   prefix by at least five percentage points. Use a diverse one-prompt-per-line
   text file with `--prompt-file prompts.txt` before treating this as evidence.

8. Read the selected expert's `hidden` and `intermediate` values from step 5,
   then run E2 with those dimensions:

   ```bash
   python -m scripts.e2_microbenchmark --hidden-size 2048 --intermediate-size 1024 \
     --block-widths 32 64 128 256 512 --repetitions 100
   ```

   The CSV reports one-block transfer/compute latency and the whole-expert
   reference. A candidate is promising only when the number of blocks needed
   by E1 is faster than a full-expert transfer at the same total byte budget.

## Go/no-go rule

Continue only when all three results hold: E0 is exact within the declared
tolerance with 100% top-1 agreement, E1 has a material ordered-over-random
lift at 25% or 50%, and E2 identifies a block width whose aggregate latency is
lower than whole-expert transfer. Expect this first pass to take roughly
12–20 T4 GPU-hours, including setup and diagnosis.
