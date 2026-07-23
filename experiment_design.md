# ByteMoE: Full Experimental Design

**Working title:** *ByteMoE: Risk-Controlled Progressive Weight Streaming for Memory-Constrained Mixture-of-Experts Inference*  
**Document type:** Execution-ready research protocol  
**Version:** 1.0  
**Date:** 24 July 2026  
**Primary target:** MLSys-level systems research  
**Development hardware:** Google Colab NVIDIA T4, 16 GB VRAM  
**Status:** Proposed; novelty and feasibility must be validated experimentally

---

## 1. Executive summary

ByteMoE investigates whether a memory-constrained GPU can serve a sparse Mixture-of-Experts language model more efficiently by loading **parts of selected experts progressively**, rather than transferring every selected expert as one indivisible weight object.

The central mechanism is an exact additive decomposition of each gated feed-forward expert into neuron blocks. A small set of high-value blocks can remain resident on the GPU, while the remaining blocks are stored in CPU memory or another storage tier. During inference, ByteMoE estimates which additional block is most likely to improve the model output per unit of transfer time, streams blocks asynchronously, and stops when a calibrated policy predicts that further blocks are unlikely to change the full-model decision beyond a user-specified risk level.

The primary research claim to test is:

> **Progressive, token-adaptive streaming within routed experts can reduce host-to-device weight traffic and decode latency more effectively than whole-expert offloading, caching, static compression, or fixed neuron dropping, while maintaining a calibrated level of agreement with full-model inference.**

The project does **not** claim that a T4 physically stores or executes all parameters of a trillion-parameter model. Real experiments will use open MoE models at manageable scale. A trace-driven simulator, calibrated and validated against real measurements, will be used only to study larger hypothetical configurations. All trillion-scale results must be labeled as projections.

A top-tier paper is plausible only if the study demonstrates all of the following:

1. A real wall-clock improvement, not only fewer theoretical FLOPs.
2. A Pareto improvement over strong contemporary baselines.
3. Generalization across at least two MoE architectures and two hardware configurations.
4. A statistically valid risk-control procedure.
5. A clear novelty boundary against expert caching, prefetching, pruning, quantization, fine-grained offloading, and early-exit risk control.

---

## 2. Research positioning and novelty boundary

### 2.1 Existing problem

Sparse MoE models activate only a subset of experts for each token, reducing active computation relative to total model capacity. However, all expert weights still require storage. When expert weights do not fit in GPU memory, systems move them from CPU RAM or storage during inference. This can make decode bandwidth-bound and create long per-token stalls.

Existing approaches include:

- whole-expert caching and prefetching;
- request-level expert activation tracing;
- CPU execution of selected experts;
- cache-aware rerouting;
- mixed-precision expert quantization;
- expert pruning, slimming, merging, and SVD compression;
- fine-grained expert offloading;
- neuron-level computation dropping;
- speculative prediction of future expert choices;
- risk-controlled early exits from layers or reasoning traces.

Representative primary sources are listed in Section 24.

### 2.2 Proposed novelty

ByteMoE combines four properties:

1. **Exact within-expert additive partitioning:** all blocks together exactly reproduce the original expert computation, subject only to the numerical precision used by the implementation.
2. **Progressive weight materialization:** selected expert weights are not transferred atomically; useful partial expert outputs can be computed as blocks arrive.
3. **Byte-level global scheduling:** the runtime allocates transfer bandwidth to individual expert blocks according to predicted reduction in output risk per unit latency.
4. **Calibrated stopping:** an independently calibrated policy selects how many blocks to load while controlling expected disagreement with a full-model reference.

The novelty is not any one component alone. It is the treatment of **off-device expert bytes as an adaptive token-level compute resource**.

### 2.3 Claims that must not be made without evidence

Do not claim:

- “the first” until the literature search is repeated immediately before submission;
- that the T4 runs a full trillion-parameter model;
- theoretical risk guarantees under arbitrary distribution shift;
- energy savings based solely on transferred-byte reductions;
- latency gains derived only from simulations;
- architecture-independent benefits based on one model;
- exact model-output preservation when blocks are omitted.

---

## 3. Research questions

### RQ1 — Prefix usefulness

Can a small subset of an expert’s neuron blocks recover enough of the complete expert output to support useful language-model inference?

### RQ2 — Adaptive allocation

Does token-dependent block selection outperform a fixed percentage of every selected expert at the same average transferred-byte budget?

### RQ3 — Systems benefit

Does progressive block streaming reduce real decode latency after accounting for transfer granularity, kernel-launch overhead, synchronization, and scheduler cost?

### RQ4 — Risk control

Can a held-out calibration procedure select a policy that controls expected next-token disagreement with the full model at a target level such as 1%?

### RQ5 — Robustness

Does the policy retain its efficiency and calibration across domains, prompt lengths, batch sizes, and routing patterns?

### RQ6 — Scaling

Can a trace-driven simulator calibrated on real hardware accurately reproduce held-out measured configurations and then support defensible projections to larger sparse models?

---

## 4. Hypotheses

### H1 — Concentrated expert contribution

For a routed expert invocation, the contribution of intermediate neurons to the expert output is sufficiently concentrated that an ordered subset of blocks can approximate the expert output substantially better than a random subset of equal size.

**Null H1₀:** Importance-ordered blocks do not improve reconstruction or downstream fidelity over random ordering at equal byte budgets.

### H2 — Token-adaptive benefit

A token-conditioned scheduler achieves higher full-model agreement or lower perplexity than fixed-prefix streaming at equal average transferred bytes.

**Null H2₀:** Token-conditioned scheduling offers no improvement over a global fixed-prefix policy.

### H3 — End-to-end latency benefit

ByteMoE reduces median and tail decode latency relative to whole-expert on-demand offloading and standard expert caching under constrained GPU memory.

**Null H3₀:** Reduced transfer volume is offset by small-copy inefficiency, scheduling overhead, or fragmented computation.

### H4 — Calibrated risk

On exchangeable test samples from the calibration distribution, the selected policy’s expected next-token disagreement with the full reference remains below a target risk level with the chosen confidence.

**Null H4₀:** Observed disagreement exceeds the calibrated upper bound or the procedure fails to find a useful policy.

### H5 — Cross-model transferability

The method improves the quality–latency Pareto frontier on at least two different MoE architectures.

**Null H5₀:** Improvements are specific to one model’s expert structure or routing behavior.

### H6 — Simulator validity

A simulator fitted to a subset of measured block sizes, bandwidths, and cache budgets predicts held-out latency and transfer volume with sufficiently low error to justify sensitivity analysis.

**Null H6₀:** Simulation error is too high or systematically biased for credible scale projections.

---

## 5. Mathematical formulation

### 5.1 Standard gated expert

For token hidden state \(x \in \mathbb{R}^{d}\), a common SwiGLU-style expert can be written as:

\[
E_e(x)=W^{(e)}_{d}\left[\operatorname{SiLU}\left(W^{(e)}_{g}x\right)\odot\left(W^{(e)}_{u}x\right)\right],
\]

where:

- \(W_g^{(e)} \in \mathbb{R}^{h\times d}\) is the gate projection;
- \(W_u^{(e)} \in \mathbb{R}^{h\times d}\) is the up projection;
- \(W_d^{(e)} \in \mathbb{R}^{d\times h}\) is the down projection;
- \(h\) is the expert intermediate width.

Let

\[
z_i(x)=\operatorname{SiLU}\left(w^{(e)}_{g,i}x\right)\left(w^{(e)}_{u,i}x\right),
\]

where \(w_{g,i}\) and \(w_{u,i}\) are rows corresponding to intermediate neuron \(i\). Then:

\[
E_e(x)=\sum_{i=1}^{h} W^{(e)}_{d,:,i}z_i(x).
\]

This sum permits an exact additive partition.

### 5.2 Exact neuron-block decomposition

Partition the intermediate neuron indices into disjoint blocks:

\[
\mathcal{B}_e=\{B_{e,1},B_{e,2},\dots,B_{e,M}\},
\quad
\bigcup_{j=1}^{M}B_{e,j}=\{1,\dots,h\}.
\]

Define block output:

\[
E_{e,j}(x)=W^{(e)}_{d,:,B_{e,j}}
\left[
\operatorname{SiLU}\left(W^{(e)}_{g,B_{e,j},:}x\right)
\odot
W^{(e)}_{u,B_{e,j},:}x
\right].
\]

The complete expert is exactly:

\[
E_e(x)=\sum_{j=1}^{M}E_{e,j}(x).
\]

A partial expert computation using block subset \(S_e(x)\) is:

\[
\widetilde{E}_e(x;S_e)=\sum_{j\in S_e(x)}E_{e,j}(x).
\]

When \(S_e=\{1,\dots,M\}\), the original expert is recovered exactly, apart from floating-point accumulation order.

### 5.3 Routed MoE layer

For top-\(k\) selected experts \(\mathcal{A}(x)\) and router weights \(r_e(x)\):

\[
Y(x)=\sum_{e\in\mathcal{A}(x)}r_e(x)E_e(x).
\]

ByteMoE approximates it as:

\[
\widetilde{Y}(x)=
\sum_{e\in\mathcal{A}(x)}r_e(x)
\sum_{j\in S_e(x)}E_{e,j}(x).
\]

### 5.4 Scheduling objective

For candidate block \((e,j)\), let:

- \(b_{e,j}\): transferred bytes;
- \(t^{\mathrm{copy}}_{e,j}\): copy time;
- \(t^{\mathrm{compute}}_{e,j}\): compute time;
- \(t^{\mathrm{overlap}}_{e,j}\): predicted overlap;
- \(\widehat{\Delta R}_{e,j}\): predicted reduction in disagreement risk or output error.

A basic priority score is:

\[
P_{e,j}=
\frac{\widehat{\Delta R}_{e,j}}
{t^{\mathrm{copy}}_{e,j}+t^{\mathrm{compute}}_{e,j}-t^{\mathrm{overlap}}_{e,j}+\epsilon}.
\]

The scheduler streams the available block with the greatest score, subject to dependency, memory, and synchronization constraints.

### 5.5 Risk definition

For full-reference next-token prediction \(y^*(q)\) and ByteMoE prediction \(\hat y_{\pi}(q)\) under policy \(\pi\), define bounded disagreement loss:

\[
\ell(q;\pi)=\mathbf{1}\left[\hat y_{\pi}(q)\neq y^*(q)\right].
\]

The expected disagreement risk is:

\[
R(\pi)=\mathbb{E}[\ell(q;\pi)].
\]

Given target \(\alpha\), the calibration stage selects the lowest-cost policy whose upper confidence bound on risk is at most \(\alpha\).

Sequence-level and semantic risks will be reported separately and will not be conflated with next-token agreement.

---

## 6. ByteMoE system design

### 6.1 Components

1. **Reference model:** frozen original MoE model.
2. **Block packer:** partitions expert neurons into storage-aligned blocks.
3. **Resident core:** selected blocks kept in GPU memory.
4. **Importance sketch:** lightweight GPU-resident predictor estimating block value from the token hidden state and routing context.
5. **Block cache:** GPU cache for recently or frequently useful non-core blocks.
6. **Asynchronous transfer engine:** copies blocks from pinned CPU memory using dedicated CUDA streams.
7. **Partial expert kernels:** compute and accumulate a block output as soon as its weights arrive.
8. **Global scheduler:** orders block transfers across routed experts and layers.
9. **Stopping policy:** determines when the allocated blocks are sufficient.
10. **Profiler and trace logger:** records routing, transfer, computation, cache, and output-fidelity data.

### 6.2 Resident core

The resident core is a fixed collection of blocks chosen under a GPU-memory budget. Candidate selection strategies:

- highest global activation-weighted contribution;
- highest average output norm;
- highest task-agnostic reconstruction gain;
- highest frequency-weighted gain;
- knapsack optimization using gain per byte;
- layer-specific allocation based on sensitivity.

The core must be selected using only training/calibration data, never the final test set.

### 6.3 Block construction

Test these constructions:

#### C1 — Contiguous blocks

Divide neurons by original index. This is the simplest systems baseline and preserves contiguous storage.

#### C2 — Importance-sorted blocks

Rank neurons using calibration statistics, then group adjacent ranks into blocks. Store a permuted packed representation to retain contiguous transfers.

#### C3 — Balanced clustered blocks

Cluster neurons using activation co-occurrence or output-vector similarity, then balance clusters to equal byte sizes.

#### C4 — Hardware-aware blocks

Jointly optimize contribution concentration and transfer efficiency with block sizes constrained to multiples suitable for DMA and GEMM kernels.

The primary paper should use C4 if it yields real gains; C1–C3 are ablations.

### 6.4 Importance targets

For an invocation \((x,e,j)\), possible supervised targets include:

1. local block-output norm:
   \[
   v^{\mathrm{norm}}_{e,j}(x)=\|r_e(x)E_{e,j}(x)\|_2;
   \]
2. reduction in expert reconstruction error;
3. reduction in MoE-layer output error;
4. reduction in final-logit KL divergence;
5. probability that including the block changes the full-reference top-1 token;
6. loss reduction per transferred byte.

The main scheduler target should be the most downstream signal that remains cheap and stable to learn. A hierarchical approach is recommended:

- predict local contribution for all candidate blocks;
- use a smaller correction model to map local signals to final-logit risk.

### 6.5 Importance sketch models

Compare:

- linear predictor from hidden state;
- low-rank bilinear predictor conditioned on expert and block embeddings;
- two-layer MLP;
- shared predictor with learned expert/block embeddings;
- random-projection sketch plus linear head;
- oracle target computed from complete blocks.

The predictor’s GPU memory and compute must be included in all measurements.

### 6.6 Runtime modes

#### Mode A — Fixed prefix

Load the first \(m\) blocks for every routed expert. This is a required baseline.

#### Mode B — Token-adaptive prefix

Choose \(m_e(x)\) independently for each routed expert using predicted difficulty.

#### Mode C — Global block scheduling

Select blocks across all currently active experts by marginal value per latency.

#### Mode D — Cache-aware global scheduling

Include cache residency and eviction cost in the priority score.

#### Mode E — Risk-controlled scheduling

Apply the calibrated threshold or policy parameter selected on held-out data.

### 6.7 Cache policy

Compare:

- no non-core cache;
- LRU;
- LFU;
- score-aware eviction;
- GreedyDual-size-style value normalized by bytes;
- oracle future-use cache.

The core allocation and dynamic cache allocation must be reported separately.

### 6.8 Asynchronous pipeline

For each layer during decode:

1. compute router logits and selected experts;
2. execute resident expert blocks;
3. score off-device candidate blocks;
4. enqueue highest-priority transfers to a copy stream;
5. execute arrived blocks on a compute stream;
6. accumulate partial expert outputs;
7. stop according to policy or continue until full expert completion;
8. release or cache blocks according to the cache policy;
9. record a structured trace.

The implementation must explicitly measure synchronization overhead and transfer–compute overlap.

---

## 7. Models

### 7.1 Model A — Controlled TinyMoE

Purpose: method development, multi-seed experiments, and controlled architecture studies.

Recommended target:

- 150M–500M total parameters;
- 12–16 transformer layers;
- 8 or 16 experts per MoE layer;
- top-2 routing;
- intermediate expert width divisible into 8–64 blocks;
- context length 1,024–2,048;
- dense or shared attention backbone.

Two options:

1. train from scratch on a modest open corpus;
2. upcycle a small dense checkpoint into an MoE and fine-tune.

A minimum of five random seeds is required for claims about learned partitioning or predictor training on this model.

### 7.2 Model B — OLMoE-1B-7B

OLMoE-1B-7B is the primary open-model target because it has approximately 7B total parameters, approximately 1B active parameters per token, 64 experts, top-8 routing, and open weights/code/training artifacts [R1]. Its many routed experts provide a useful stress test for block selection and memory movement.

Use both base and instruction-tuned variants only if the implementation budget permits. The base model is sufficient for perplexity and token-agreement experiments; the instruction variant is useful for downstream generation tasks.

### 7.3 Model C — Secondary architecture

At least one architecturally different MoE should be included for a top-tier submission:

- Mixtral 8×7B, top-2 routing [R2];
- DeepSeekMoE 16B or a smaller DeepSeekMoE-compatible model with fine-grained/shared experts [R3];
- another open model available at submission time.

This phase will likely require hardware beyond a free Colab T4. A limited but real secondary-hardware experiment is preferable to a broad simulation-only claim.

### 7.4 Model eligibility rules

A model is eligible when:

- expert FFNs can be instrumented and repacked;
- routing decisions can be logged;
- licenses permit research evaluation and release of derived code;
- a full-reference execution path can be produced;
- evaluation can be repeated deterministically enough for paired comparison.

---

## 8. Data design

Use disjoint splits for decomposition statistics, predictor training, policy calibration, and final evaluation.

### 8.1 Partition D1 — Block-statistics corpus

Purpose: estimate neuron/block importance and activation co-occurrence.

Recommended size:

- TinyMoE: 5M–20M tokens;
- OLMoE: 1M–5M tokens, depending on cost.

Content mixture:

- general web text;
- Wikipedia-style text;
- code;
- mathematical or scientific text;
- conversational/instruction text.

### 8.2 Partition D2 — Sketch-training corpus

Purpose: train block-value predictors.

Recommended target:

- at least 100,000 routed expert invocations;
- stratified by layer, expert frequency, sequence position, and domain;
- include rare experts through controlled oversampling.

### 8.3 Partition D3 — Risk-calibration corpus

Purpose: select policy thresholds and compute risk upper bounds.

Recommended target:

- 10,000–50,000 next-token decisions;
- prompts and token positions disjoint from D1 and D2;
- a fixed generation protocol;
- no reuse for hyperparameter tuning after calibration.

### 8.4 Partition D4 — In-distribution test corpus

Purpose: final quality and risk evaluation on a matched distribution.

Recommended target:

- at least 20,000 next-token decisions;
- at least 1,000 complete generation examples;
- separated by prompt and document, not merely by token.

### 8.5 Partition D5 — Distribution-shift test corpus

Purpose: evaluate failure of calibration and adaptive efficiency.

Domains:

- source code;
- mathematics;
- multilingual text;
- long-form conversation;
- scientific abstracts;
- adversarially repetitive text;
- random or synthetically unusual token patterns.

No formal in-distribution guarantee should be claimed for D5. Report empirical risk and calibration drift.

### 8.6 Evaluation task suite

Recommended tasks, subject to model compatibility and licensing:

- WikiText-103 or another standard language-modeling set for perplexity;
- HellaSwag;
- ARC-Challenge;
- MMLU subsets or full MMLU;
- GSM8K;
- HumanEval or MBPP for code;
- TruthfulQA or another generation-oriented benchmark;
- a long-context continuation set;
- an internally constructed routing-stress set with no use in quality claims.

Report exact task versions, prompts, few-shot settings, decoding parameters, and evaluation harness commit hashes.

---

## 9. Hardware and software environments

### 9.1 H1 — Colab T4 development system

Record for every run:

- GPU name and memory;
- CUDA driver and runtime versions;
- PyTorch version;
- Transformers version;
- CPU model and core count;
- available and used system RAM;
- pinned-memory limit;
- storage type and available space;
- measured host-to-device bandwidth;
- thermal or clock behavior where observable.

Do not assume all Colab T4 sessions have identical host hardware.

### 9.2 H2 — Second consumer system

Recommended:

- RTX 3090/4090 or similar 24 GB GPU;
- known PCIe generation;
- at least 64 GB CPU RAM.

Purpose: validate transfer granularity and determine whether ByteMoE benefits persist with higher bandwidth and compute.

### 9.3 H3 — Datacenter system

Recommended if accessible:

- A100, L40S, H100, or comparable GPU;
- controlled CPU and PCIe/NVLink configuration.

A top-tier paper does not necessarily require many datacenter GPUs, but one controlled datacenter comparison would materially strengthen external validity.

### 9.4 Software stack

Initial stack:

- Python 3.11;
- PyTorch 2.x;
- CUDA-compatible T4 build;
- Hugging Face Transformers;
- Accelerate;
- bitsandbytes where useful for reference storage;
- lm-evaluation-harness or Ai2 evaluation tooling;
- Triton or custom CUDA extension for optimized block kernels;
- pandas/Polars and Parquet for traces;
- Hydra or structured YAML configuration;
- pytest for correctness tests.

Freeze exact versions before final experiments.

---

## 10. Baselines

### 10.1 Mandatory systems baselines

#### B0 — Full-reference inference

All expert blocks are executed. Weights may be resident or offloaded depending on hardware, but outputs define the reference.

#### B1 — Whole-expert on-demand offloading

Transfer every selected expert in full when absent from GPU memory.

#### B2 — Whole-expert LRU cache

Keep complete recently used experts within the same dynamic memory budget.

#### B3 — Whole-expert LFU cache

Cache complete frequently used experts.

#### B4 — Trace/prediction prefetch baseline

Prefetch complete experts using recent routing or a lightweight predictor, inspired by systems such as MoE-Infinity and newer expert-speculation work [R4, R7].

#### B5 — CPU expert execution

Execute absent experts on CPU rather than transfer them, inspired by Fiddler [R5]. Include only if a fair implementation is achievable.

#### B6 — Fixed neuron fraction

Use the same fixed block count for every selected expert and token.

#### B7 — Static GPU core only

Execute only resident blocks; never stream corrections.

### 10.2 Mandatory compression/sparsity baselines

#### B8 — Uniform quantization

Quantize all experts to a fixed bit width.

#### B9 — Mixed-precision expert quantization

Assign bit widths by expert importance, reflecting the class of approaches represented by Mixture Compressor [R9].

#### B10 — SVD-based expert compression

Implement or reuse MoE-SVD when compatible [R8].

#### B11 — Static neuron dropping

Drop low-importance neurons or blocks without progressive streaming.

#### B12 — Dynamic neuron dropping

Activation-aware dynamic dropping, approximating the relevant comparison to neuron-level sparse approaches such as DualSparse-MoE [R10].

### 10.3 ByteMoE variants

#### V1 — ByteMoE fixed prefix

Importance-ordered blocks with fixed prefix length.

#### V2 — ByteMoE token-adaptive

Independent token-conditioned block budgets.

#### V3 — ByteMoE global scheduler

Block-level value-per-latency scheduling.

#### V4 — ByteMoE cache-aware scheduler

Global scheduling plus block cache.

#### V5 — ByteMoE risk-controlled

V4 with threshold selected on D3.

### 10.4 Oracle baselines

#### O1 — Oracle block ranking

Rank blocks using actual full-reference downstream effect.

#### O2 — Oracle stopping budget

Choose the smallest block subset that produces the same top-1 next token as the full model.

#### O3 — Oracle cache

Use future trace knowledge for eviction.

Oracles define headroom and should never be compared as deployable systems.

---

## 11. Metrics

### 11.1 Quality and fidelity

- next-token top-1 agreement with full reference;
- top-5 set agreement;
- KL divergence between output distributions;
- Jensen–Shannon divergence;
- change in negative log-likelihood;
- perplexity;
- sequence-level exact agreement;
- normalized edit distance between generations;
- downstream benchmark score;
- pass@1 for code where applicable;
- accuracy difference from full reference;
- calibration risk and upper confidence bound.

### 11.2 Systems

- time to first token;
- time per output token;
- tokens per second;
- p50, p90, p95, and p99 latency;
- GPU peak allocated and reserved memory;
- CPU peak memory;
- bytes transferred host-to-device per token;
- bytes transferred device-to-host per token;
- number of transfers per token;
- mean and distribution of transfer sizes;
- effective copy bandwidth;
- transfer–compute overlap percentage;
- cache hit rate by block and by byte;
- cache eviction count;
- scheduler runtime;
- kernel-launch count;
- GPU utilization;
- CPU utilization;
- SSD bytes read if SSD is evaluated;
- energy per token only when measured with a defensible method.

### 11.3 Predictor and scheduler

- block-value prediction correlation;
- top-\(k\) block recall;
- pairwise ranking accuracy;
- normalized discounted cumulative gain;
- risk-reduction prediction error;
- wasted-prefetch bytes;
- useful-prefetch bytes;
- average blocks loaded by layer, expert, domain, and token type;
- fraction of tokens using only the resident core;
- fraction requiring complete experts.

### 11.4 Simulator

- mean absolute percentage error for latency;
- error for transfer volume;
- error for cache hit rate;
- error across unseen block sizes;
- calibration plots measured versus predicted;
- sensitivity to bandwidth, latency, and compute assumptions.

---

## 12. Experimental protocol

## Experiment E0 — Correctness and exact reconstruction

### Goal

Verify that the packed block representation reconstructs the original expert and model outputs when all blocks are loaded.

### Procedure

1. Select 100 random sequences and 1,000 hidden states per model.
2. Run the unmodified full expert.
3. Run all packed blocks and accumulate their outputs.
4. Compare expert outputs, layer outputs, logits, and generated tokens.
5. Repeat for FP32 reference, FP16/BF16 where supported, and the intended quantized storage mode.
6. Test different block orders and accumulation orders.

### Metrics

- maximum absolute error;
- relative \(L_2\) error;
- logit error;
- top-1 agreement;
- deterministic generation agreement.

### Pass criterion

All-block ByteMoE must match the unmodified model within a predeclared numerical tolerance and produce 100% top-1 agreement on the correctness set. If it does not, no downstream experiment may call it an exact reconstruction.

---

## Experiment E1 — Is progressive partial execution viable?

### Goal

Determine whether small block subsets approximate full expert outputs meaningfully.

### Independent variables

- block construction C1–C4;
- block count \(M \in \{4,8,16,32,64\}\);
- loaded byte fraction \(f \in \{0.05,0.10,0.20,0.30,0.50,0.75,1.0\}\);
- ordering strategy: random, output norm, activation frequency, reconstruction gain, learned global ordering;
- model and layer.

### Procedure

1. Compute complete expert outputs on D4 hidden states.
2. Evaluate partial sums at each budget.
3. Measure local expert error, layer error, logit error, and next-token agreement.
4. Stratify results by router weight, expert frequency, layer depth, and domain.
5. Compare paired examples across orderings.

### Main figures

- expert-output error versus byte fraction;
- next-token agreement versus byte fraction;
- distributions of the minimum fraction required for full-token agreement;
- layer-wise sensitivity heatmap.

### Success criterion

At least one non-oracle ordering must substantially dominate random ordering and show a broad distribution of required budgets. If nearly all tokens require almost all blocks, the central premise is weak.

---

## Experiment E2 — Block granularity and hardware efficiency

### Goal

Find block sizes that balance adaptability and transfer/kernel efficiency.

### Independent variables

- block payload from approximately 16 KB to 16 MB;
- contiguous versus reordered packing;
- pinned versus pageable CPU memory;
- synchronous versus asynchronous copy;
- one versus multiple CUDA streams;
- fused versus separate block kernels;
- batch size \(1,2,4,8\).

### Procedure

1. Build synthetic expert blocks matching real model dimensions.
2. Measure copy time, effective bandwidth, kernel time, launch overhead, and overlap.
3. Repeat at least 100 times after warm-up.
4. Fit piecewise latency models.
5. Validate using real expert blocks.

### Outputs

- minimum efficient transfer size;
- optimal block-size range for T4;
- hardware-specific constraints used by C4;
- latency-model parameters for the simulator.

### Success criterion

Identify at least one granularity where progressive loading does not lose most of its theoretical byte advantage to overhead.

---

## Experiment E3 — Importance-sketch accuracy

### Goal

Test whether a small resident predictor can rank useful off-device blocks.

### Independent variables

- predictor architecture;
- predictor size;
- target definition;
- shared versus layer-specific predictor;
- training-domain composition;
- quantization of predictor weights.

### Procedure

1. Generate labels by executing complete blocks on D2.
2. Split D2 by document and prompt into train/validation subsets.
3. Train predictors without changing the base model.
4. Evaluate on D4 and D5.
5. Report predictor cost separately.

### Metrics

- Spearman correlation;
- top-\(k\) recall;
- ranking regret relative to oracle;
- bytes required to reach a target output error;
- added latency and memory.

### Success criterion

The learned ranking must improve the quality-per-byte curve over the best global ordering after including predictor overhead.

---

## Experiment E4 — Scheduling policy comparison

### Goal

Compare fixed, local adaptive, and global block allocation.

### Policies

- equal blocks per selected expert;
- allocation proportional to router weight;
- greedy predicted output-norm reduction;
- greedy final-risk reduction;
- value per byte;
- value per predicted latency;
- cache-aware value per latency;
- oracle.

### Budgets

Evaluate equal average transfer budgets spanning 10%–100% of full selected-expert bytes.

### Metrics

- next-token agreement;
- perplexity change;
- transfer bytes;
- block count;
- real latency;
- scheduler cost;
- regret versus oracle.

### Main result

A Pareto frontier of quality versus real time and quality versus transferred bytes.

### Success criterion

The global scheduler must dominate fixed-prefix execution on a meaningful part of the frontier. If improvement appears only in bytes but not real time, the systems claim must be narrowed.

---

## Experiment E5 — Risk calibration

### Goal

Select deployable policies that meet target disagreement risks.

### Policy family

Create a finite set \(\Pi=\{\pi_1,\dots,\pi_K\}\) by varying:

- stopping threshold;
- minimum and maximum blocks;
- cache-aware score weight;
- conservative fallback rule.

### Calibration method

1. Freeze all model, predictor, scheduler, and policy-family choices before D3.
2. For each policy, compute binary next-token disagreement against the full reference.
3. Calculate a one-sided finite-sample upper confidence bound for each policy’s risk.
4. Correct for selecting among multiple policies using a predeclared family-wise method or a Learn-then-Test-style procedure.
5. Among policies with upper bound \(\leq\alpha\), choose the one with minimum mean latency or transferred bytes.
6. Evaluate once on D4.

### Target risks

\[
\alpha \in \{0.001,0.005,0.01,0.02,0.05\}.
\]

### Report

- target risk;
- empirical calibration risk;
- risk upper bound;
- test risk;
- coverage of the claimed bound;
- latency and bytes saved;
- no-solution cases where no efficient policy satisfies the target.

### Distribution shift

Repeat the selected policy on D5 but describe results as empirical robustness, not guaranteed risk control.

### Success criterion

At least one useful policy must satisfy a stringent target such as 1% next-token disagreement on D4 while delivering a meaningful systems improvement.

---

## Experiment E6 — End-to-end T4 evaluation

### Goal

Demonstrate real interactive decoding improvements on the development platform.

### Workloads

- batch size 1 interactive chatbot;
- batch sizes 2, 4, and 8;
- prompt lengths 128, 512, 2,048, and maximum feasible length;
- output lengths 32, 128, and 256;
- general, code, math, and conversational prompts.

### Memory regimes

Artificially cap dynamic expert GPU memory to:

- resident core only;
- 5%;
- 10%;
- 20%;
- 40% of total expert-weight bytes;
- largest feasible reference point.

### Baselines

At minimum B1–B4, B6–B9, B11–B12, and V1–V5.

### Timing protocol

1. Restart or clear state according to a predeclared cold/warm policy.
2. Perform warm-up runs that are not included.
3. Use CUDA events for device timing and wall-clock timers for end-to-end latency.
4. Synchronize only where required for correct measurement.
5. Run at least 30 repeated generations per configuration, using paired prompts.
6. Randomize configuration order to reduce thermal and session-order bias.
7. Record raw per-token traces.

### Primary endpoint

Time per output token at batch size 1 under a fixed GPU expert-memory budget and a calibrated 1% next-token disagreement target.

### Secondary endpoints

- p95 token latency;
- transferred bytes per token;
- full-model task-score difference;
- energy if reliable measurement is available.

### Target result, not guaranteed

A strong outcome would be:

- at least 20% reduction in median time per output token;
- at least 30% reduction in p95 token latency;
- at least 40% reduction in expert-weight transfer bytes;
- observed risk no greater than the calibrated target;
- less than 0.5 absolute percentage-point average task degradation.

---

## Experiment E7 — Cross-hardware validation

### Goal

Determine whether ByteMoE is a general method or a T4-specific optimization.

### Procedure

Repeat the core E2, E5, and E6 configurations on H2 and, if possible, H3.

Do not retune policy thresholds on the test set. Two valid reporting modes:

1. hardware-specific calibration using a separate calibration partition;
2. direct transfer of the T4 policy, reported explicitly as transfer.

### Analysis

- relative speedup versus bandwidth;
- change in optimal block size;
- scheduler choices by hardware;
- quality invariance;
- whether faster links reduce the method’s advantage.

### Success criterion

ByteMoE should improve at least one relevant constrained-memory regime on the second hardware platform. It need not win when all weights fit in GPU memory.

---

## Experiment E8 — Cross-model validation

### Goal

Validate on a second MoE architecture.

### Procedure

1. Re-run E0 and E1 to verify additive partitioning and prefix viability.
2. Repack experts using the same algorithm, not manually selected model-specific blocks.
3. Train or adapt only the sketch module and calibration policy.
4. Run a focused E6 comparison under one or two memory budgets.

### Success criterion

The method must show a consistent quality–latency benefit, even if the magnitude differs. A negative result should be analyzed using routing locality, number of active experts, intermediate width, and block contribution concentration.

---

## Experiment E9 — Ablation study

Remove or replace one component at a time:

- no importance ordering;
- no learned sketch;
- no global scheduling;
- no cache awareness;
- no asynchronous copy;
- no resident core;
- no risk calibration;
- fixed threshold across hardware;
- random block partition;
- equal block sizes versus hardware-aware sizes;
- no router-weight feature;
- no final-risk target, local norm only;
- full precision versus quantized blocks;
- no fallback to full expert.

Report both quality and wall-clock effects. The final method should not be a collection of components whose individual contributions are negligible.

---

## Experiment E10 — Distribution-shift and failure analysis

### Goal

Find where ByteMoE fails.

### Stressors

- domain shift;
- rare experts;
- long sequences;
- abrupt topic change;
- repetitive routing;
- adversarially low logit margins;
- random hidden-state perturbations;
- reduced CPU–GPU bandwidth;
- increased copy latency;
- memory pressure from large KV cache;
- batch mixing of unrelated domains.

### Outputs

- risk drift;
- policy overconfidence;
- blocks required per token;
- tail-latency spikes;
- experts/layers responsible for failures;
- fallback frequency.

### Safety mechanism

Implement a conservative fallback that loads all blocks when:

- the stopping score is outside the calibrated support;
- predictor confidence is below a threshold;
- a distribution-shift detector triggers;
- a configured maximum approximation risk is exceeded.

The fallback’s cost must be included in reported averages.

---

## Experiment E11 — Trace-driven scale simulation

### Goal

Study large sparse-model configurations without claiming physical execution.

### Simulator inputs

- measured transfer latency curves;
- block compute time distributions;
- cache lookup and eviction overhead;
- scheduler overhead;
- routing traces;
- block-value distributions;
- CPU RAM, SSD, or network bandwidth models;
- expert count, expert width, top-\(k\), layers, and batch size.

### Validation

1. Fit simulator parameters using only a subset of E2/E6 configurations.
2. Predict held-out block sizes, cache budgets, and workload lengths.
3. Require low prediction error before extrapolation.
4. Report confidence or sensitivity bands rather than a single precise number.

### Scale scenarios

- 100B total / 10B active;
- 500B total / 20B active;
- 1T total / 30B–50B active;
- expert storage in CPU DDR;
- expert storage in NVMe;
- remote expert storage over configurable network links.

### Required labels

Every figure must distinguish:

- **measured**;
- **simulated within measured scale**;
- **extrapolated beyond measured scale**.

### Prohibited conclusion

Do not state that ByteMoE “runs a 1T model on a T4.” A permissible conclusion is that the calibrated simulator projects a particular traffic or latency behavior under explicit assumptions.

---

## 13. Statistical analysis plan

### 13.1 Experimental unit

- For token-level fidelity: prompt/document group, not individual correlated tokens, is the preferred resampling unit.
- For generation latency: one complete generation run.
- For model-training comparisons: random seed.
- For task accuracy: benchmark example.

### 13.2 Repetitions

- kernel microbenchmarks: at least 100 timed repetitions after warm-up;
- end-to-end configurations: at least 30 paired runs;
- TinyMoE learned-method comparisons: at least five seeds;
- large-model predictor training: at least three seeds where feasible;
- downstream generation: enough examples for useful confidence intervals, preferably at least 500 per task.

### 13.3 Confidence intervals

Use:

- paired bootstrap confidence intervals for latency and quality differences;
- cluster bootstrap by prompt/document for token metrics;
- exact or conservative binomial bounds for disagreement risk;
- seed-level t intervals or nonparametric alternatives when seed count is small;
- bootstrap intervals for Pareto-frontier differences.

### 13.4 Hypothesis tests

Predeclare a small number of primary comparisons:

1. V5 versus B1 on TPOT;
2. V5 versus B2 on TPOT;
3. V5 versus B9/B10 on quality at matched memory or bytes;
4. V5 test risk versus target \(\alpha\);
5. V3 versus V1 at matched bytes.

Use paired tests where possible. Correct secondary-comparison multiplicity with Holm’s method or report them as exploratory.

### 13.5 Effect sizes

Report:

- absolute milliseconds and relative speedup;
- absolute bytes and percentage reduction;
- absolute task-score difference;
- risk difference;
- standardized effect sizes only where meaningful.

### 13.6 Non-inferiority margins

Predeclare margins separately by metric. Suggested starting targets:

- average downstream accuracy degradation: no more than 0.5 absolute percentage point;
- perplexity increase: no more than 1% relative;
- next-token disagreement: according to calibrated \(\alpha\);
- sequence semantic quality: task-specific and reported without overstating equivalence.

Margins must be justified from application requirements rather than selected after seeing results.

---

## 14. Detailed risk-control protocol

### 14.1 Why next-token agreement is the primary risk

It is:

- bounded and measurable;
- available without human labels;
- directly comparable with a frozen full reference;
- suitable for large calibration sets.

It does not prove semantic equivalence over complete generations, so sequence metrics remain necessary.

### 14.2 Policy family construction

Before calibration, define policies parameterized by threshold \(\tau\):

\[
\pi_\tau(x)=\text{continue loading while }s(x,S)<\tau,
\]

where \(s\) estimates stability or confidence of the current partial computation.

Candidate features:

- logit margin from a lightweight look-ahead approximation;
- change in partial expert output after the latest block;
- predicted norm of unseen blocks;
- router entropy and selected-expert weights;
- layer index;
- cache state;
- cumulative bytes;
- predictor uncertainty;
- token frequency and sequence position.

### 14.3 Calibration integrity

The calibration set must not be used to:

- redesign features;
- change the policy family;
- tune the sketch network;
- choose benchmark tasks;
- alter the target risk after examining results.

If any such change occurs, create a new untouched calibration split.

### 14.4 Guarantee statement

A careful statement is:

> Under the calibration procedure’s assumptions and for samples exchangeable with the calibration distribution, the selected policy is designed to keep the expected bounded disagreement loss below the target with the stated confidence.

Do not state per-token certainty or robustness to arbitrary domain shift.

---

## 15. Implementation plan

### 15.1 Repository structure

```text
bytemoe/
├── README.md
├── pyproject.toml
├── requirements-lock.txt
├── configs/
│   ├── models/
│   ├── hardware/
│   ├── experiments/
│   └── policies/
├── bytemoe/
│   ├── models/
│   │   ├── adapters.py
│   │   ├── expert_partition.py
│   │   └── reference.py
│   ├── packing/
│   │   ├── block_packer.py
│   │   ├── layouts.py
│   │   └── manifest.py
│   ├── runtime/
│   │   ├── cache.py
│   │   ├── transfer.py
│   │   ├── scheduler.py
│   │   ├── stopping.py
│   │   └── engine.py
│   ├── predictors/
│   │   ├── datasets.py
│   │   ├── models.py
│   │   └── training.py
│   ├── kernels/
│   │   ├── torch_reference.py
│   │   ├── triton_blocks.py
│   │   └── cuda_extension/
│   ├── calibration/
│   │   ├── risk_bounds.py
│   │   └── policy_selection.py
│   ├── tracing/
│   │   ├── schema.py
│   │   ├── hooks.py
│   │   └── profiler.py
│   └── simulation/
│       ├── latency_model.py
│       ├── simulator.py
│       └── validation.py
├── scripts/
│   ├── collect_activations.py
│   ├── pack_experts.py
│   ├── train_sketch.py
│   ├── calibrate_policy.py
│   ├── run_benchmark.py
│   └── run_simulation.py
├── tests/
│   ├── test_exact_reconstruction.py
│   ├── test_packing.py
│   ├── test_cache.py
│   ├── test_scheduler.py
│   └── test_risk_bounds.py
├── notebooks/
└── results/
    ├── raw/
    ├── processed/
    ├── figures/
    └── manifests/
```

### 15.2 Packed block manifest

Each block record should contain:

```text
model_id
model_revision
layer_id
expert_id
block_id
neuron_indices
storage_offset
storage_length_bytes
dtype_or_quantization
checksum
resident_core_flag
importance_statistics
packing_version
```

### 15.3 Trace schema

Each token-layer-expert-block event should record:

```text
run_id
prompt_id
token_position
layer_id
expert_id
router_rank
router_weight
block_id
resident_before
cache_hit
transfer_start_ns
transfer_end_ns
compute_start_ns
compute_end_ns
bytes_transferred
predicted_value
oracle_value_if_collected
selected_or_skipped
stop_reason
gpu_memory_bytes
```

Final-token records should include logits/fidelity metrics and complete latency.

### 15.4 Reference pseudocode

```python
for layer in model.layers:
    x = layer.pre_moe(x)
    experts, router_weights = layer.route(x)

    partial = zeros_like(x)
    candidates = []

    for expert, weight in zip(experts, router_weights):
        for block in resident_blocks(expert):
            partial += weight * execute_block(block, x)

        for block in off_device_blocks(expert):
            score = value_predictor(x, layer, expert, block, weight)
            candidates.append((score, expert, block, weight))

    while not stop_policy.should_stop(partial, candidates, runtime_state):
        item = scheduler.select(candidates, cache_state, runtime_state)
        block = transfer_engine.fetch_async(item.block)
        wait_until_ready(block)
        partial += item.weight * execute_block(block, x)
        scheduler.update_after_execution(item, partial)

    x = layer.post_moe(x, partial)
```

The optimized implementation should avoid serial waits and should overlap scoring, transfer, and computation.

---

## 16. Reproducibility controls

For every result, save:

- Git commit hash;
- model repository and immutable revision;
- dataset revision and split hashes;
- exact prompt templates;
- random seeds;
- environment lock file;
- hardware profile;
- packed-block manifest and checksums;
- calibration policy file;
- raw traces;
- command line and configuration;
- warm-up and timing protocol;
- failure logs.

Release:

- correctness tests;
- trace-processing scripts;
- figure-generation scripts;
- simulator validation data;
- negative results where possible.

Never hand-copy headline numbers into figures. Generate tables and plots from stored result files.

---

## 17. Go/no-go gates

### Gate G0 — Exactness

**Go:** all-block execution matches reference within tolerance.  
**No-go:** packing changes model behavior materially.

### Gate G1 — Prefix viability

**Go:** importance ordering substantially outperforms random ordering and many tokens need less than the full expert.  
**Pivot:** if not, test learned residual blocks or narrow to specific layers/models.  
**Stop:** if useful partial outputs are absent across models.

### Gate G2 — Hardware viability

**Go:** efficient block sizes permit useful transfer reduction after overhead.  
**Pivot:** use fewer, larger blocks or batched multi-token transfers.  
**Stop:** if whole-expert transfers are always faster at relevant budgets.

### Gate G3 — Predictor value

**Go:** learned token-dependent ranking beats global ordering after including its cost.  
**Pivot:** use simpler layer-specific budgets or router-weight allocation.

### Gate G4 — Risk-controlled utility

**Go:** at least one calibrated policy meets \(\alpha=1\%\) with meaningful savings.  
**Pivot:** use 2% or 5% risk and report the trade-off; do not hide failure at 1%.

### Gate G5 — End-to-end result

**Go for top-tier submission:** clear Pareto dominance on two models or hardware platforms.  
**Workshop-level result:** strong T4 result on one model plus careful analysis.  
**Negative-result paper:** clear evidence explaining why progressive expert streaming fails.

---

## 18. Failure modes and pivots

### F1 — Importance is not concentrated

**Observation:** partial blocks produce large errors until nearly 100% of bytes are loaded.

**Pivots:**

- learn an exact invertible neuron permutation that concentrates contribution;
- retrain/fine-tune experts with prefix-dropout while preserving a full-reference path;
- focus on selected layers where concentration exists;
- switch from neuron blocks to low-rank residual blocks, explicitly giving up exact reconstruction at intermediate stages.

### F2 — Small transfers are inefficient

**Pivots:**

- batch blocks across experts and tokens;
- use double buffering;
- enlarge blocks;
- schedule one large contiguous transfer containing several selected blocks;
- perform CPU computation for tiny residual blocks.

### F3 — Predictor overhead is too high

**Pivots:**

- global layer-specific ordering;
- small linear sketches;
- quantized predictor;
- reuse router logits and hidden-state norms;
- update decisions only every several tokens.

### F4 — Approximation errors compound across layers

**Pivots:**

- allocate larger budgets to sensitive layers;
- enforce periodic full-expert refresh tokens;
- calibrate layer-specific risk budgets;
- use conservative completion in late layers.

### F5 — Risk calibration fails under shift

**Pivots:**

- detect shift and fall back to full experts;
- maintain domain-specific calibration profiles;
- online recalibration with delayed reference samples;
- report empirical robustness without a guarantee.

### F6 — Baselines are stronger

This is scientifically valuable. Analyze whether their advantage comes from:

- high routing locality;
- low top-\(k\);
- large transfer blocks;
- strong quantization tolerance;
- CPU execution efficiency;
- inability of partial experts to preserve logits.

Do not change metrics or baselines after seeing unfavorable results.

---

## 19. Publication success criteria

### 19.1 Minimum credible paper

- exact decomposition implementation;
- real T4 end-to-end prototype;
- OLMoE evaluation;
- strong whole-expert and fixed-prefix baselines;
- calibrated risk experiment;
- open traces and code;
- honest limitation of scale claims.

### 19.2 Strong workshop or specialized venue

- meaningful T4 speedup;
- multiple tasks and domains;
- comprehensive ablations;
- one secondary hardware configuration;
- simulator validated on held-out measurements.

### 19.3 Top-tier target

The study should ideally show:

- two or more MoE architectures;
- two or more distinct hardware configurations;
- real Pareto improvement over contemporary offloading, prediction, compression, and neuron-dropping baselines;
- statistically valid risk control;
- optimized kernels rather than only Python hooks;
- clear analysis of when the method helps or fails;
- full reproducibility package;
- no dependence on trillion-scale simulation for the main contribution.

Acceptance cannot be guaranteed. The experimental evidence, not the concept alone, determines venue viability.

---

## 20. Recommended execution timeline

### Phase 1 — Weeks 1–2: literature and feasibility lock

- repeat novelty search;
- select model revisions;
- implement expert hooks;
- inspect expert tensor layouts;
- define exact packing format;
- run E0 on one layer.

### Phase 2 — Weeks 3–5: reference implementation

- implement full block decomposition;
- write correctness tests;
- collect activation traces;
- run E1 with PyTorch reference code;
- make G1 decision.

### Phase 3 — Weeks 6–8: systems microbenchmarks

- pinned-memory transfer engine;
- asynchronous streams;
- block kernels;
- run E2;
- select hardware-aware block sizes.

### Phase 4 — Weeks 9–12: predictor and scheduler

- build D2;
- train sketch predictors;
- implement fixed and adaptive schedulers;
- run E3 and E4.

### Phase 5 — Weeks 13–15: risk control

- freeze policy family;
- prepare D3;
- implement bounds and policy selection;
- run E5 once after freezing choices.

### Phase 6 — Weeks 16–18: T4 end-to-end study

- run E6;
- execute ablations;
- profile bottlenecks;
- optimize the critical path;
- rerun frozen final matrix.

### Phase 7 — Weeks 19–21: external validity

- acquire second hardware;
- run E7;
- port to second model;
- run E8.

### Phase 8 — Weeks 22–24: simulation and paper

- validate simulator;
- run E11 projections;
- finalize statistical analysis;
- write paper and artifact appendix;
- perform final literature/patent search;
- release reproducibility package.

---

## 21. Expected paper figures and tables

### Figures

1. ByteMoE architecture and asynchronous timeline.
2. Exact neuron-block decomposition diagram.
3. Quality versus expert-byte fraction.
4. Distribution of per-token minimum required bytes.
5. Predictor ranking accuracy and oracle gap.
6. Quality–bytes Pareto frontier.
7. Quality–latency Pareto frontier.
8. TPOT and p95 latency across memory budgets.
9. Calibration target versus observed risk.
10. Layer-by-layer block allocation heatmap.
11. Transfer–compute overlap timeline.
12. Cross-hardware speedup versus measured bandwidth.
13. Distribution-shift failure analysis.
14. Simulator measured-versus-predicted validation.
15. Clearly labeled large-scale sensitivity projection.

### Tables

1. Model architectures and expert configurations.
2. Hardware/software environments.
3. Baseline definitions and implementation sources.
4. Main end-to-end results.
5. Downstream quality results.
6. Risk-control results.
7. Ablation results.
8. Predictor overhead.
9. Simulator error.
10. Negative results and failure regimes.

---

## 22. Paper outline

1. **Introduction**
   - bandwidth bottleneck;
   - atomic expert transfer limitation;
   - ByteMoE insight;
   - contributions.
2. **Background and motivation**
   - MoE routing;
   - offloading;
   - measured block-contribution concentration;
   - limitations of existing approaches.
3. **Exact progressive expert representation**
   - additive SwiGLU partition;
   - packing and resident core.
4. **Byte-level scheduling**
   - value prediction;
   - latency-aware scheduling;
   - cache integration.
5. **Risk-controlled stopping**
   - policy family;
   - calibration protocol;
   - guarantee scope.
6. **Implementation**
   - kernels;
   - asynchronous pipeline;
   - memory management.
7. **Evaluation methodology**
   - models, tasks, hardware, baselines, statistics.
8. **Results**
   - fidelity;
   - end-to-end performance;
   - calibration;
   - ablations;
   - cross-hardware/model.
9. **Scale study**
   - simulator validation first;
   - projections second.
10. **Limitations and failure cases**
11. **Related work**
12. **Conclusion**

---

## 23. Research integrity checklist

Before submission confirm:

- [ ] All headline improvements are wall-clock, not only FLOP estimates.
- [ ] Baselines use equal memory, quality, or transfer constraints where claimed.
- [ ] The full-reference path is clearly defined.
- [ ] Calibration data are disjoint from development and final testing.
- [ ] Risk statements specify assumptions and confidence.
- [ ] Simulation is visually distinguished from measurement.
- [ ] Trillion-scale language is limited to projections.
- [ ] Results include confidence intervals and raw-run counts.
- [ ] Negative configurations are not silently removed.
- [ ] Predictor and scheduler overhead are included.
- [ ] Cache warm/cold conditions are stated.
- [ ] Model and dataset versions are immutable and cited.
- [ ] The novelty search is updated before submission.
- [ ] “First” claims are independently verified or removed.
- [ ] Code reproduces tables and figures from raw traces.

---

## 24. Primary references and novelty threats

### [R1] OLMoE

Muennighoff et al., “OLMoE: Open Mixture-of-Experts Language Models,” 2024.  
https://arxiv.org/abs/2409.02060

Relevance: primary open model; approximately 7B total and 1B active parameters; open weights, code, and training artifacts.

### [R2] Mixtral

Jiang et al., “Mixtral of Experts,” 2024.  
https://arxiv.org/abs/2401.04088

Relevance: widely used top-2 sparse MoE and secondary architecture candidate.

### [R3] DeepSeekMoE

Dai et al., “DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models,” 2024.  
https://arxiv.org/abs/2401.06066

Relevance: fine-grained expert segmentation and shared experts; important architectural comparison.

### [R4] MoE-Infinity

Xue et al., “MoE-Infinity: Offloading-Efficient MoE Model Serving,” 2024.  
https://arxiv.org/abs/2401.14361

Novelty threat: request-level expert tracing, prefetching, and caching.

### [R5] Fiddler

Kamahori et al., “Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models,” 2024.  
https://arxiv.org/abs/2402.07033

Novelty threat: CPU execution as an alternative to weight movement.

### [R6] fMoE

Yu et al., “fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving,” 2025.  
https://arxiv.org/abs/2502.05370

Novelty threat: fine-grained expert offloading, prefetching, caching, and semantic hints. ByteMoE must distinguish within-expert progressive execution from generic fine-grained placement.

### [R7] Speculating Experts

Madan et al., “Speculating Experts Accelerates Inference for Mixture-of-Experts,” 2026.  
https://arxiv.org/abs/2603.19289

Novelty threat: prediction of future experts and overlapping transfers with computation.

### [R8] MoE-SVD

Li et al., “MoE-SVD: Structured Mixture-of-Experts LLMs Compression via Singular Value Decomposition,” ICML 2025.  
https://openreview.net/forum?id=acJ3vdFljk

Novelty threat: expert decomposition, shared matrices, and structured compression.

### [R9] Mixture Compressor

Huang et al., “Mixture Compressor for Mixture-of-Experts LLMs Gains More,” ICLR 2025.  
https://openreview.net/forum?id=hheFYjOsWO

Novelty threat: mixed-precision expert quantization and online dynamic expert pruning.

### [R10] DualSparse-MoE

Cai et al., “DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction,” 2025.  
https://arxiv.org/abs/2508.18376

Novelty threat: dynamic tensor-level dropping and static neuron-level reconstruction.

### [R11] Cache-Conditional Experts

Skliar et al., “Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference,” TMLR 2025.  
https://openreview.net/forum?id=ul4W26KEKz

Novelty threat: cache-aware changes to routing for constrained devices.

### [R12] Risk-controlled early exit

Jazbec et al., “Fast yet Safe: Early-Exiting with Risk Control,” NeurIPS 2024.  
https://openreview.net/forum?id=hACHuDzi1U

Novelty threat: risk control for adaptive computation and early exit. ByteMoE must claim a different controlled object: progressive off-device expert-weight materialization.

### [R13] Risk control for reasoning budgets

Wang et al., “Conformal Thinking: Risk Control for Reasoning on a Compute Budget,” ICML 2026.  
https://openreview.net/forum?id=noDJPmA3ha

Novelty threat: calibrated allocation of variable computation budgets in LLM reasoning.

### [R14] Consumer/edge MoE inference study

Alfarizy et al., “Does Mixture-of-Experts Actually Help Inference on Consumer and Edge Hardware? An Empirical Study,” 2026.  
https://arxiv.org/abs/2606.21428

Relevance: supports the need to evaluate real memory bandwidth, dispatch, and energy rather than relying on active-parameter FLOPs.

### [R15] Holistic MoE compression study

He et al., “Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques,” TMLR 2025.  
https://openreview.net/forum?id=HTpMOl6xSI

Novelty threat: expert slimming/trimming and larger structural dropping strategies.

---

## 25. Final decision rule

Proceed toward a top-tier submission only when the evidence supports this narrow claim:

> ByteMoE turns routed expert weights into a progressively materialized resource and, through hardware-aware byte scheduling plus calibrated stopping, improves the measured quality–latency frontier under constrained GPU memory.

If the method reduces bytes but not wall-clock latency, publish it only as an analysis or compression result. If it improves one model on one T4 session but fails to generalize, target a workshop and describe the boundary honestly. If it produces consistent real gains across models and hardware while meeting calibrated risk targets, the project has a credible top-tier systems story.
