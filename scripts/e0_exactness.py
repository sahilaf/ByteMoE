from __future__ import annotations

import argparse

import torch

from bytemoe.blocks import block_indices, execute_blocks, execute_expert, static_importance_order
from bytemoe.hf_adapter import choose_expert, print_experts, replacement_hook
from scripts.common import DEFAULT_MODEL, load_model, load_prompts, prompt_batches, write_json


def max_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = (reference.float() - candidate.float()).abs()
    relative_l2 = difference.norm() / reference.float().norm().clamp_min(1e-12)
    return {"max_abs_error": difference.max().item(), "relative_l2_error": relative_l2.item()}


def logits_for(model, batches):
    logits = []
    with torch.inference_mode():
        for batch in batches:
            output = model(**batch).logits
            # Gather the next-token distribution at each sequence's final
            # non-padding token so batches with different prompt lengths remain
            # comparable.
            positions = batch["attention_mask"].sum(dim=1).sub(1)
            logits.append(output[torch.arange(output.shape[0], device=output.device), positions].detach().float().cpu())
    return torch.cat(logits, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: exact SwiGLU block reconstruction gate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--expert-index", type=int, default=0)
    parser.add_argument("--expert-contains")
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-copies", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--output", default="results/e0_exactness.json")
    parser.add_argument("--list-experts", action="store_true")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model, args.dtype)
    if args.list_experts:
        print_experts(model)
        return
    expert = choose_expert(model, args.expert_contains, args.expert_index)
    order = static_importance_order(expert.weights).to(device)
    blocks = block_indices(order, args.blocks)
    prompts = load_prompts(args.prompt_file, args.prompt_copies)
    batches = list(prompt_batches(tokenizer, device, prompts, args.batch_size, args.max_length))

    # Direct expert arithmetic test on actual routed inputs, captured by a hook.
    captured: list[torch.Tensor] = []
    capture = expert.module.register_forward_hook(lambda _m, inputs, _o: captured.append(inputs[0].detach()))
    reference_logits = logits_for(model, batches)
    capture.remove()
    if not captured:
        raise RuntimeError(f"Expert {expert.name} was not routed for these prompts; choose another expert or add prompts.")
    # Captured activations were created under inference_mode by `logits_for`.
    # Keep the reconstruction arithmetic in the same mode: inference tensors
    # cannot be used by an autograd-tracked operation outside this context.
    direct_errors = []
    with torch.inference_mode():
        for hidden in captured:
            full = execute_expert(hidden, expert.weights)
            rebuilt = execute_blocks(hidden, expert.weights, blocks)
            direct_errors.append(max_metrics(full, rebuilt))

    # End-to-end verification: replace that expert by the sum of all packed blocks.
    hook = expert.module.register_forward_hook(replacement_hook(expert.weights, lambda x, w: execute_blocks(x, w, blocks)))
    rebuilt_logits = logits_for(model, batches)
    hook.remove()
    logit_metrics = max_metrics(reference_logits, rebuilt_logits)
    top1_agreement = (reference_logits.argmax(-1) == rebuilt_logits.argmax(-1)).float().mean().item()
    max_direct = max(item["max_abs_error"] for item in direct_errors)
    passed = max_direct <= args.tolerance and top1_agreement == 1.0
    report = {
        "experiment": "E0", "model": args.model, "expert": expert.name, "blocks": args.blocks,
        "dtype": args.dtype, "prompt_count": len(prompts), "direct_max_abs_error": max_direct,
        "direct_max_relative_l2_error": max(item["relative_l2_error"] for item in direct_errors),
        "logit": logit_metrics, "top1_agreement": top1_agreement, "tolerance": args.tolerance, "passed": passed,
    }
    write_json(args.output, report)
    print(report)
    if not passed:
        raise SystemExit("E0 FAILED: do not run approximate experiments until this is fixed.")


if __name__ == "__main__":
    main()
