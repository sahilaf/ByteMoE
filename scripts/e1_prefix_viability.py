from __future__ import annotations

import argparse

import torch

from bytemoe.blocks import execute_expert, random_order, static_importance_order
from bytemoe.hf_adapter import choose_expert, find_swiglu_experts, replacement_hook
from scripts.common import DEFAULT_MODEL, load_model, load_prompts, prompt_batches, write_json


def logits_for(model, batches):
    output = []
    with torch.inference_mode():
        for batch in batches:
            logits = model(**batch).logits
            positions = batch["attention_mask"].sum(dim=1).sub(1)
            output.append(logits[torch.arange(logits.shape[0], device=logits.device), positions].detach().float().cpu())
    return torch.cat(output, dim=0)


def choose_active_expert(model, batches, contains: str | None):
    candidates = find_swiglu_experts(model)
    if contains:
        candidates = [expert for expert in candidates if contains in expert.name]
    if not candidates:
        raise RuntimeError("No eligible SwiGLU experts were found.")
    token_counts = [0] * len(candidates)
    handles = []
    for index, candidate in enumerate(candidates):
        def count_tokens(_module, inputs, _output, index=index):
            if inputs and isinstance(inputs[0], torch.Tensor):
                token_counts[index] += inputs[0].shape[0]
        handles.append(candidate.module.register_forward_hook(count_tokens))
    logits_for(model, batches)
    for handle in handles:
        handle.remove()
    best_index = max(range(len(candidates)), key=token_counts.__getitem__)
    if token_counts[best_index] == 0:
        raise RuntimeError("No routed expert received tokens for the supplied prompts.")
    expert = candidates[best_index]
    print(f"Auto-selected active expert [{best_index}] {expert.name} ({token_counts[best_index]} routed tokens).")
    return expert


def score_prefix(model, expert, order, fraction, reference_logits, batches):
    count = max(1, round(expert.weights.intermediate_size * fraction))
    indices = order[:count]
    hook = expert.module.register_forward_hook(
        replacement_hook(expert.weights, lambda x, weights: execute_expert(x, weights, indices.to(x.device)))
    )
    candidate_logits = logits_for(model, batches)
    hook.remove()
    difference = reference_logits - candidate_logits
    return {
        "fraction": fraction,
        "neurons": count,
        "top1_agreement": (reference_logits.argmax(-1) == candidate_logits.argmax(-1)).float().mean().item(),
        "mean_logit_l2": difference.norm(dim=-1).mean().item(),
        "mean_kl_reference_to_partial": torch.nn.functional.kl_div(
            candidate_logits.log_softmax(-1), reference_logits.softmax(-1), reduction="batchmean"
        ).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: importance-ordered prefix viability gate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--expert-index", type=int, default=-1, help="-1 automatically selects the most-active routed expert")
    parser.add_argument("--expert-contains")
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.25, 0.5))
    parser.add_argument("--random-seeds", type=int, nargs="+", default=(11, 29, 47))
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-copies", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-agreement-lift", type=float, default=0.05)
    parser.add_argument("--output", default="results/e1_prefix_viability.json")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model, args.dtype)
    prompts = load_prompts(args.prompt_file, args.prompt_copies)
    batches = list(prompt_batches(tokenizer, device, prompts, args.batch_size, args.max_length))
    expert = (
        choose_active_expert(model, batches, args.expert_contains)
        if args.expert_index == -1
        else choose_expert(model, args.expert_contains, args.expert_index)
    )
    reference_logits = logits_for(model, batches)
    importance = static_importance_order(expert.weights).to(device)
    results = []
    for fraction in args.fractions:
        ordered = score_prefix(model, expert, importance, fraction, reference_logits, batches)
        random_scores = [
            score_prefix(model, expert, random_order(expert.weights.intermediate_size, device, seed), fraction, reference_logits, batches)
            for seed in args.random_seeds
        ]
        random_mean = sum(item["top1_agreement"] for item in random_scores) / len(random_scores)
        ordered["random_mean_top1_agreement"] = random_mean
        ordered["agreement_lift_over_random"] = ordered["top1_agreement"] - random_mean
        ordered["random_trials"] = random_scores
        results.append(ordered)
    passed = any(item["agreement_lift_over_random"] >= args.min_agreement_lift for item in results)
    report = {
        "experiment": "E1", "model": args.model, "expert": expert.name, "dtype": args.dtype,
        "prompt_count": len(prompts), "min_agreement_lift": args.min_agreement_lift,
        "results": results, "passed": passed,
        "note": "This quick gate approximates one routed expert. Scale to all experts/layers only after it passes.",
    }
    write_json(args.output, report)
    print(report)
    if not passed:
        print("E1 did not meet the predeclared top-1 lift criterion; results were saved for analysis.")


if __name__ == "__main__":
    main()
