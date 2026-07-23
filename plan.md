# ByteMoE Execution Plan

## Purpose and operating rules

This plan converts `experiment_design.md` into an ordered implementation and
evaluation program. The initial target is a reproducible T4 prototype; claims
about a second model, second hardware platform, or large-scale simulation are
made only after the preceding gates pass.

- Keep D1 (statistics), D2 (predictor training), D3 (calibration), D4
  (in-distribution test), and D5 (shift test) disjoint by document/prompt.
- Store every run's immutable model and dataset revisions, configuration,
  seed, environment, hardware profile, manifest checksum, raw trace, and
  command line.
- Freeze implementation and policy choices before consuming D3; do not tune
  against D4 or D5.
- Treat all latency claims as wall-clock claims backed by raw traces. Label
  simulator results distinctly from measurements.
- Start with PyTorch reference implementations. Optimize only validated hot
  paths, preserving a tested reference path.

## Milestone overview

| Milestone | Weeks | Outcome | Gate |
|---|---:|---|---|
| M0 | 1–2 | Environment, model choice, and exact interface specification | G0-ready |
| M1 | 3–5 | Correct block packing and prefix-viability evidence | G0, G1 |
| M2 | 6–8 | Measured transfer/kernel constraints and async runtime | G2 |
| M3 | 9–12 | Predictor and adaptive scheduler | G3 |
| M4 | 13–15 | Frozen, risk-controlled policy | G4 |
| M5 | 16–18 | Full T4 study and ablations | G5 (T4) |
| M6 | 19–21 | Second hardware and second-model evidence | G5 (external) |
| M7 | 22–24 | Validated simulator, paper artifacts, and release package | Submission-ready |

## M0 — Feasibility lock and reproducible foundation (Weeks 1–2)

### Step 0.1 — Lock scope and literature boundary

1. Repeat the novelty search for within-expert streaming, fine-grained MoE
   offloading, expert prediction/prefetching, neuron sparsity, compression,
   and risk-controlled adaptive computation.
2. Record the search date, queries, primary sources, and a claim-to-prior-work
   comparison table in `docs/related_work.md`.
3. Freeze the initial narrow claim: progressive materialization of routed
   expert blocks improves a measured quality–latency frontier under constrained
   GPU expert memory.
4. Explicitly exclude unsupported claims: “first”, trillion-parameter T4
   execution, energy savings without measurement, and shift-robust guarantees.

**Deliverable:** literature/claim register with named novelty threats.

### Step 0.2 — Select and validate model targets

1. Select a TinyMoE for rapid multi-seed development and lock its architecture,
   checkpoint/training recipe, tokenizer, and license.
2. Select a pinned OLMoE revision as the primary real-model target.
3. Identify one eligible secondary architecture and document the hardware
   required to run its full-reference path.
4. Inspect expert modules, router outputs, tensor layouts, dtypes, and
   supported generation paths for every selected model.
5. Write model adapters that expose: routed expert IDs and weights, expert
   SwiGLU projections, layer input/output hooks, and a deterministic reference
   forward path.

**Acceptance:** each adapter can log routing decisions and reproduce an
unmodified model's logits for a small fixed prompt set.

### Step 0.3 — Establish project structure and environment

1. Create the package structure described in design §15.1: adapters,
   partitioning, packing, runtime, predictors, kernels, calibration, tracing,
   simulation, scripts, tests, configs, and results directories.
2. Add a Python 3.11 environment definition and lock exact versions for
   PyTorch, CUDA-compatible dependencies, Transformers, Accelerate, test,
   configuration, and trace-processing libraries.
3. Add a configuration schema with immutable model/dataset revisions, hardware
   profile, seed, memory budget, block layout, policy, and timing protocol.
4. Add a run manifest writer and a single command that records environment,
   git commit, GPU/CPU/RAM details, storage, CUDA/PyTorch versions, and measured
   host-to-device bandwidth.
5. Add CI/local tests for config validation, deterministic fixed prompts, and
   manifest creation.

**Deliverable:** a reproducible smoke run with a complete run manifest.

### Step 0.4 — Define data partitions before experimentation

1. Choose compatible data sources and fixed revisions for general text, code,
   math/science, conversational text, and evaluation tasks.
2. Generate document-level D1–D5 split manifests with hashes; retain no prompt
   or document overlap between D1/D2/D3/D4/D5.
3. Prepare D1 at the planned scale, D2 with at least 100,000 stratified routed
   expert invocations, D3 with 10,000–50,000 decisions, and D4 with at least
   20,000 decisions plus 1,000 generations, subject to model cost.
4. Define D5 stress subsets for code, math, multilingual, long-context,
   scientific, repetitive, and synthetic-unusual prompts.
5. Record exact prompt templates, few-shot settings, decoding parameters, and
   evaluation harness revisions.

**Acceptance:** split validator proves document/prompt disjointness and
published hashes are included in every experiment configuration.

## M1 — Exact progressive representation and prefix viability (Weeks 3–5)

### Step 1.1 — Implement block decomposition and packing

1. Implement the exact SwiGLU neuron-block decomposition for the target expert
   layout: gate/up rows and corresponding down-projection columns.
2. Implement C1 contiguous blocks first, with configurable block counts
   \(M \in \{4,8,16,32,64\}\).
3. Implement the packed block format and manifest fields from design §15.2,
   including tensor offsets, dtypes, checksums, neuron permutation, importance
   statistics, and resident-core flag.
4. Add C2 importance-sorted packing; retain the reversible permutation and
   contiguous packed storage.
5. Add C3 clustered and C4 hardware-aware layouts only after the C1/C2
   reference path is correct.

**Tests:** packing round trip, manifest checksum validation, dimensional
validation, and all-block equivalence under every supported dtype.

### Step 1.2 — Run E0: correctness and exact reconstruction

1. Sample 100 sequences and 1,000 hidden states per supported model.
2. Compare original expert output with all packed-block sums in FP32, intended
   compute dtype, intended storage dtype, and intended quantized mode.
3. Repeat across block orders and accumulation orders.
4. Compare expert output, MoE-layer output, logits, top-1 predictions, and
   deterministic generations.
5. Predeclare tolerances by dtype and save per-case error distributions.

**Gate G0:** all-block execution is within tolerance and has 100% top-1
agreement on the correctness set. If it fails, stop all approximate experiments
and fix packing/reference discrepancies.

### Step 1.3 — Build tracing and block-statistics pipeline

1. Implement structured tracing following design §15.3, with one event per
   token/layer/expert/block and a final-token fidelity/latency record.
2. Collect D1 routes, hidden states (or safely serialized feature sketches),
   block outputs, router weights, activation frequency, output norms, and
   reconstruction gains.
3. Compute global and per-layer/expert importance statistics without using D3–D5.
4. Select candidate resident cores using gain-per-byte, frequency-weighted gain,
   and layer-sensitive allocation.
5. Version the trace schema and write trace validation/aggregation tests.

### Step 1.4 — Run E1: progressive partial-execution viability

1. On D4 hidden states, produce complete reference expert and model outputs.
2. Evaluate C1–C4 where available at all planned block counts, byte fractions,
   and orderings (random, norm, frequency, reconstruction gain, learned global).
3. Stratify metrics by model, layer, router weight, expert frequency, depth,
   and domain.
4. Generate error/byte, agreement/byte, minimum-required-byte distribution,
   and layer-sensitivity figures from stored data.

**Gate G1:** at least one deployable ordering clearly beats random ordering and
many tokens need materially less than all expert bytes. Otherwise pivot to
larger residual-style blocks, selected layers, or report a negative feasibility
result before investing in systems optimization.

## M2 — Hardware-aware runtime and microbenchmarks (Weeks 6–8)

### Step 2.1 — Establish baseline runtime paths

1. Implement B0 full reference, B1 whole-expert on-demand transfer, B2 LRU,
   B3 LFU, B6 fixed neuron fraction, and B7 core-only paths under common
   memory-budget accounting.
2. Add B4 whole-expert prefetch only once its predictor and warm/cold policy
   are specified. Add B5 CPU execution only if it can be measured fairly.
3. Instrument bytes, cache events, transfer events, kernel events, memory,
   scheduler time, and end-to-end token timing identically in all paths.

### Step 2.2 — Run E2: find viable block granularity

1. Benchmark synthetic and real blocks from approximately 16 KB to 16 MB.
2. Cross pinned/pageable CPU memory, contiguous/reordered layouts, sync/async
   copies, one/multiple CUDA streams, fused/separate kernels, and batches
   1/2/4/8.
3. Warm up, then collect at least 100 timed repetitions for each configuration.
4. Measure copy/compute/launch times, effective bandwidth, synchronization,
   and transfer–compute overlap using CUDA events plus end-to-end clocks.
5. Fit and save a piecewise latency model; reserve held-out configurations for
   its later simulator validation.

**Gate G2:** select a T4-efficient block-size range that retains meaningful
byte savings after transfer and launch overhead. If not, batch blocks/tokens,
increase block size, double-buffer, or narrow the method's systems claim.

### Step 2.3 — Implement asynchronous reference engine

1. Place packed nonresident blocks in pinned CPU memory and make residency,
   transfer ownership, and eviction explicit.
2. Implement a copy stream, compute stream, events, buffer lifecycle, and
   deterministic synchronous fallback.
3. Execute resident blocks first; score candidates; enqueue transfers; execute
   arrived blocks; accumulate partial expert outputs; then cache or release.
4. Ensure no accidental serial `wait` exists in the optimized path and verify
   numerical equivalence when all blocks are completed.
5. Record overlap and synchronization reasons in traces.

**Acceptance:** asynchronous all-block execution is correct and produces a
trace explaining its real transfer/compute overlap.

## M3 — Importance predictor and adaptive scheduling (Weeks 9–12)

### Step 3.1 — Generate predictor data and train sketches

1. Use only D2 to label candidate blocks with local contribution, reconstruction
   gain, and selected downstream signals such as logit-KL or top-1-change risk.
2. Split D2 by document/prompt; stratify by layer, expert frequency, token
   position, router rank, and domain, oversampling rare experts where needed.
3. Train and compare linear, low-rank bilinear, small MLP, shared embedding,
   and random-projection baselines.
4. Include feature extraction, predictor memory, and predictor runtime in every
   candidate's cost.
5. Freeze a predictor family and checkpoint selection rule using D2 validation
   only.

### Step 3.2 — Run E3: predictor evaluation

1. Evaluate on D4 and D5 without fitting.
2. Report Spearman correlation, top-k recall, ranking accuracy, NDCG, oracle
   regret, bytes to target error, predictor memory, and added latency.
3. Compare against global importance ordering and router-weight-only allocation.

**Gate G3:** a learned ranking improves quality per byte over the best global
ordering after predictor overhead. Otherwise use a simpler global/layer policy
and document the negative result.

### Step 3.3 — Implement scheduler variants and cache policies

1. Implement V1 fixed prefix, V2 token-adaptive budget, V3 global scheduling,
   V4 cache-aware scheduling, and their deterministic reference modes.
2. Implement equal-per-expert, router-weight, predicted-norm, predicted-risk,
   value-per-byte, and value-per-latency policies.
3. Implement no-cache, LRU, LFU, score-aware, and byte-normalized cache
   eviction under separate core and dynamic-cache budgets.
4. Add oracle ranking, stopping, and cache modes strictly for headroom analysis.
5. Unit-test priority ties, budget limits, cache accounting, no-progress cases,
   and repeatable scheduling decisions.

### Step 3.4 — Run E4: scheduling comparison

1. Compare all deployable policies at matched average byte budgets from 10% to
   100% of full selected-expert bytes.
2. Measure agreement, perplexity delta, bytes, blocks, real latency, scheduler
   cost, cache behavior, and oracle regret.
3. Construct paired quality–bytes and quality–latency Pareto frontiers.

**Acceptance:** V3/V4 improves a meaningful frontier region over V1. If gains
appear only in bytes, retain it as an efficiency analysis rather than a latency
claim and focus optimization on the measured bottleneck.

## M4 — Risk-controlled policy (Weeks 13–15)

### Step 4.1 — Freeze the policy family before calibration

1. Freeze model revision, packing, core budget, predictor, scheduler, fallback
   rule, metric definitions, and finite policy family \(\Pi\).
2. Vary only predeclared stopping threshold, min/max blocks, cache-score weight,
   and conservative fallback settings.
3. Implement one-sided finite-sample risk upper bounds and a predeclared
   multiplicity correction (family-wise or Learn-then-Test).
4. Implement a no-solution outcome; never relax a target after observing D3.

### Step 4.2 — Run E5: calibration and one-shot D4 test

1. On D3, compute binary next-token disagreement against B0 for every policy.
2. For each \(\alpha \in \{0.001,0.005,0.01,0.02,0.05\}\), select the
   lowest-latency or lowest-byte policy whose corrected upper bound is at most
   \(\alpha\).
3. Lock selected policies and evaluate once on D4.
4. Report calibration risk, bound, D4 risk, confidence/coverage method, bytes,
   latency, and no-solution cases.
5. Run D5 only as empirical robustness and report drift without guarantees.

**Gate G4:** at least one useful policy meets a stringent target (aim: 1%) on
D4 while saving material latency or bytes. Otherwise report a transparent
risk–efficiency trade-off at 2%/5% or stop risk-control claims.

### Step 4.3 — Add conservative runtime fallback

1. Define supported-score ranges and predictor-confidence thresholds from D2/D3.
2. Trigger full completion for out-of-support scores, low confidence, detected
   shift, or configured risk limit.
3. Trace every fallback reason and include fallback cost in all aggregates.
4. Test that fallback always produces the full-reference layer result when all
   blocks complete.

## M5 — T4 end-to-end evaluation and ablations (Weeks 16–18)

### Step 5.1 — Freeze the final T4 matrix

1. Define interactive batch-1 and batch-2/4/8 workloads, prompt lengths
   128/512/2,048/max feasible, output lengths 32/128/256, and general/code/
   math/conversation prompts.
2. Define core-only, 5%, 10%, 20%, 40%, and largest-feasible expert-memory
   regimes, with KV-cache pressure recorded separately.
3. Include B1–B4, B6–B9, B11–B12, and V1–V5 where fair implementations exist;
   document omissions and why.
4. Predeclare cold/warm policy, warm-up count, paired prompt order,
   randomization, and the primary endpoint: batch-1 TPOT at fixed memory and
   calibrated 1% next-token risk.

### Step 5.2 — Run E6 and compute statistics

1. Run at least 30 paired, randomized generation repetitions per configuration
   after warm-up; save raw per-token traces.
2. Measure wall-clock end-to-end time and CUDA device events, p50/p90/p95/p99,
   TTFT, TPOT, throughput, bytes, copy sizes, cache activity, GPU/CPU memory,
   and scheduler overhead.
3. Measure fidelity, perplexity, task-score difference, and calibration risk on
   the matching quality protocol.
4. Use prompt/document-clustered and paired bootstrap intervals. Predeclare the
   primary comparisons: V5 vs B1/B2 TPOT, V3 vs V1 at matched bytes, V5 vs
   compression at matched memory/bytes, and V5 risk vs \(\alpha\).

### Step 5.3 — Run E9/E10 ablations and failure analysis

1. Ablate ordering, sketch, global scheduling, cache awareness, asynchronous
   copy, core, calibration, hardware-specific threshold, partition type,
   router features, risk target, quantization, and fallback.
2. Stress D5, rare experts, long sequences, topic shifts, repetitive routing,
   small logit margins, reduced bandwidth, increased copy latency, KV-cache
   pressure, and mixed-domain batches.
3. Attribute failures to layers/experts and report risk drift, tail latency,
   required bytes, fallback rate, and policy overconfidence.

**Gate G5 (T4):** determine honestly whether evidence supports a workshop-level
result, a negative result, or advancement to cross-hardware/model validation.

## M6 — External validity (Weeks 19–21)

### Step 6.1 — Cross-hardware E7

1. Profile H2 (and H3 if available): GPU, PCIe/NVLink, RAM, measured transfer
   curve, and software stack.
2. Re-run core E2, E5, and E6 configurations with a separate calibration split
   for hardware-specific policy selection, or explicitly report direct T4-policy
   transfer.
3. Compare speedup by bandwidth, changed optimal granularity, scheduler choices,
   and quality invariance.

**Acceptance:** ByteMoE helps in at least one constrained-memory regime on H2;
do not expect a benefit when all relevant weights fit on GPU.

### Step 6.2 — Cross-model E8

1. Use the same packing algorithm on the secondary model; do not hand-design
   favorable blocks.
2. Re-run E0/E1, train/adapt only the sketch and calibration policy, and run a
   focused E6 comparison at one or two memory budgets.
3. Analyze any negative result by routing locality, top-k, intermediate width,
   and contribution concentration.

**Acceptance:** report consistent quality–latency benefit or a clearly bounded
model-specific limitation.

## M7 — Simulator, publication artifacts, and release (Weeks 22–24)

### Step 7.1 — Build and validate E11 trace-driven simulator

1. Implement a simulator consuming routing traces, value distributions, measured
   transfer curves, block compute times, cache/scheduler overhead, and explicit
   storage/link assumptions.
2. Fit parameters only on a subset of E2/E6. Predict held-out block sizes,
   cache budgets, and workloads before inspecting measured results.
3. Report latency/traffic/cache error, calibration plots, and sensitivity bands.
4. Extrapolate only if held-out error is acceptably low; label outputs
   **measured**, **simulated within measured scale**, or **extrapolated**.
5. Limit scale conclusions to conditional projections; never claim physical T4
   execution of 100B–1T models from a simulator.

### Step 7.2 — Produce final analysis and paper assets

1. Generate all tables/figures from raw traces and versioned analysis scripts;
   never hand-copy numbers.
2. Produce architecture, decomposition, asynchronous timeline, quality/bytes,
   quality/latency, latency percentiles, risk calibration, ablation, failure,
   cross-platform, and simulator-validation figures.
3. Include negative configurations, baseline implementation sources, exact
   fairness constraints, confidence intervals, raw-run counts, and limitations.
4. Re-run literature and patent searches immediately before submission and
   remove unsupported novelty language.

### Step 7.3 — Release reproducibility package

1. Release adapters, packer, correctness tests, configs, manifests, trace
   schema, trace processors, figure scripts, simulator validation data, and
   locked environments subject to model/dataset licenses.
2. Provide a one-command smoke reproduction and documented resource estimates
   for each experiment tier.
3. Verify a clean checkout can regenerate a selected table/figure from supplied
   raw or sample traces.

## Decision checkpoints and pivots

| Gate | Evidence required | If it fails |
|---|---|---|
| G0 | All-block packed execution matches reference | Fix adapter/packing/dtype semantics; do not approximate yet. |
| G1 | Ordered prefixes beat random and often avoid full bytes | Try larger/learned blocks or selected layers; otherwise publish feasibility failure. |
| G2 | A measured efficient block granularity exists | Batch/enlarge blocks, double-buffer, or narrow systems claim. |
| G3 | Predictor beats global order after its own cost | Use simple layer/global policy and retain the result as an ablation. |
| G4 | Calibrated useful policy meets target risk | Report looser-risk trade-off or no-solution; do not overclaim guarantees. |
| G5 | Real, fair Pareto improvement beyond one narrow setup | Target workshop/analysis/negative result; do not rely on simulations. |

## Definition of done

The project is execution-complete when the repository can reproduce E0–E6 on
the primary model with immutable artifacts, reports all gate outcomes and
negative results, and produces a measured quality–latency conclusion. A
top-tier submission additionally requires validated E7/E8 evidence, a
held-out-validated simulator for scale analysis, and the full reproducibility
package.
