from __future__ import annotations

import argparse

import torch

from bytemoe.blocks import block_indices, execute_expert, static_importance_order
from bytemoe.hf_adapter import choose_expert, find_swiglu_experts, replacement_hook
from bytemoe.oracle import greedy_local_oracle
from scripts.common import DEFAULT_MODEL, load_model, load_prompts, prompt_batches, write_json


def logits_for(model, batches):
    values = []
    with torch.inference_mode():
        for batch in batches:
            logits = model(**batch).logits
            positions = batch["attention_mask"].sum(dim=1).sub(1)
            values.append(logits[torch.arange(logits.shape[0], device=logits.device), positions].detach().float().cpu())
    return torch.cat(values, dim=0)


def choose_active_expert(model, batches, contains: str | None):
    candidates = find_swiglu_experts(model)
    if contains:
        candidates = [expert for expert in candidates if contains in expert.name]
    if not candidates:
        raise RuntimeError("No eligible SwiGLU experts were found.")
    counts = [0] * len(candidates)
    handles = []
    for index, candidate in enumerate(candidates):
        def count_tokens(_module, inputs, _output, index=index):
            if inputs and isinstance(inputs[0], torch.Tensor):
                counts[index] += inputs[0].shape[0]
        handles.append(candidate.module.register_forward_hook(count_tokens))
    logits_for(model, batches)
    for handle in handles:
        handle.remove()
    best = max(range(len(candidates)), key=counts.__getitem__)
    if counts[best] == 0:
        raise RuntimeError("No routed expert received tokens for the supplied prompts.")
    expert = candidates[best]
    print(f"Auto-selected active expert [{best}] {expert.name} ({counts[best]} routed tokens).")
    return expert


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 oracle: token-adaptive local reconstruction upper bound")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--expert-index", type=int, default=-1)
    parser.add_argument("--expert-contains")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--oracle-blocks", type=int, default=5)
    parser.add_argument("--target-agreement", type=float, default=0.99)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-copies", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", default="results/e1_oracle_viability.json")
    args = parser.parse_args()
    if args.oracle_blocks > args.blocks:
        raise ValueError("--oracle-blocks cannot exceed --blocks")

    model, tokenizer, device = load_model(args.model, args.dtype)
    prompts = load_prompts(args.prompt_file, args.prompt_copies)
    batches = list(prompt_batches(tokenizer, device, prompts, args.batch_size, args.max_length))
    expert = choose_active_expert(model, batches, args.expert_contains) if args.expert_index == -1 else choose_expert(model, args.expert_contains, args.expert_index)
    blocks = block_indices(static_importance_order(expert.weights).to(device), args.blocks)
    local_errors: list[float] = []
    selection_counts = torch.zeros(args.blocks, dtype=torch.long)
    routed_tokens = 0

    def execute_oracle(x: torch.Tensor, weights):
        nonlocal routed_tokens
        outputs = torch.stack([execute_expert(x, weights, indices.to(x.device)) for indices in blocks], dim=1)
        partial, selections = greedy_local_oracle(outputs, args.oracle_blocks)
        full = outputs.sum(dim=1)
        errors = (full.float() - partial.float()).norm(dim=-1) / full.float().norm(dim=-1).clamp_min(1e-12)
        local_errors.extend(errors.detach().cpu().tolist())
        selection_counts.add_(torch.bincount(selections.detach().cpu().flatten(), minlength=args.blocks))
        routed_tokens += x.shape[0]
        return partial

    reference_logits = logits_for(model, batches)
    hook = expert.module.register_forward_hook(replacement_hook(expert.weights, execute_oracle))
    oracle_logits = logits_for(model, batches)
    hook.remove()
    delta = reference_logits - oracle_logits
    top1 = (reference_logits.argmax(dim=-1) == oracle_logits.argmax(dim=-1)).float().mean().item()
    report = {
        "experiment": "E1_oracle_local_reconstruction",
        "oracle_scope": "Token-specific greedy local expert-output reconstruction; upper bound only, not a final-logit or deployable oracle.",
        "model": args.model,
        "expert": expert.name,
        "prompt_count": len(prompts),
        "routed_tokens": routed_tokens,
        "candidate_blocks": args.blocks,
        "selected_blocks_per_token": args.oracle_blocks,
        "selected_fraction": args.oracle_blocks / args.blocks,
        "mean_local_relative_l2_error": sum(local_errors) / len(local_errors),
        "block_selection_count": selection_counts.tolist(),
        "top1_agreement": top1,
        "mean_logit_l2": delta.norm(dim=-1).mean().item(),
        "mean_kl_reference_to_oracle": torch.nn.functional.kl_div(oracle_logits.log_softmax(-1), reference_logits.softmax(-1), reduction="batchmean").item(),
        "target_agreement": args.target_agreement,
        "passed": top1 >= args.target_agreement,
    }
    write_json(args.output, report)
    print(report)
    if not report["passed"]:
        print("Oracle did not meet the target agreement; results were saved for analysis.")


if __name__ == "__main__":
    main()
