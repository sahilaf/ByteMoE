from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .blocks import SwiGLUWeights


@dataclass(frozen=True)
class LocatedExpert:
    name: str
    module: nn.Module
    weights: SwiGLUWeights


def _linear_weight(module: nn.Module, names: Iterable[str]) -> torch.Tensor | None:
    for name in names:
        candidate = getattr(module, name, None)
        weight = getattr(candidate, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return weight
    return None


def find_swiglu_experts(model: nn.Module) -> list[LocatedExpert]:
    """Find expert modules exposing Hugging Face-style gate/up/down projections.

    This covers common MoE implementations where each routed expert is a module.
    The command prints discovered paths so an unsupported model fails visibly,
    rather than silently benchmarking a different module.
    """
    located: list[LocatedExpert] = []
    for name, module in model.named_modules():
        gate = _linear_weight(module, ("gate_proj", "w1"))
        up = _linear_weight(module, ("up_proj", "w3"))
        down = _linear_weight(module, ("down_proj", "w2"))
        if gate is None or up is None or down is None:
            continue
        if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
            continue
        located.append(LocatedExpert(name, module, SwiGLUWeights(gate, up, down)))
    return located


def choose_expert(model: nn.Module, contains: str | None, index: int) -> LocatedExpert:
    experts = find_swiglu_experts(model)
    if contains:
        experts = [expert for expert in experts if contains in expert.name]
    if not experts:
        raise RuntimeError(
            "No SwiGLU expert module was found. Run with --list-experts, then "
            "set --expert-contains to a listed path, or add a model adapter."
        )
    if index < 0 or index >= len(experts):
        preview = "\n".join(f"  [{i}] {item.name}" for i, item in enumerate(experts[:30]))
        raise IndexError(f"expert-index {index} is out of range (found {len(experts)}).\n{preview}")
    return experts[index]


def print_experts(model: nn.Module) -> None:
    experts = find_swiglu_experts(model)
    if not experts:
        print("No module-style SwiGLU experts found.")
        return
    for i, expert in enumerate(experts):
        print(
            f"[{i}] {expert.name}: hidden={expert.weights.hidden_size}, "
            f"intermediate={expert.weights.intermediate_size}"
        )


def replacement_hook(weights: SwiGLUWeights, executor):
    """Return a forward hook that substitutes an expert's Tensor output.

    The hook is deliberately strict: the quick gate is valid only when the
    selected expert accepts one Tensor and returns one Tensor.
    """
    def hook(_module: nn.Module, args: tuple[object, ...], output: object):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "Selected expert has a non-standard forward signature. Add a small "
                "adapter for this model before interpreting E0/E1 results."
            )
        replacement = executor(args[0], weights)
        if replacement.shape != output.shape:
            raise RuntimeError(
                f"Replacement shape {tuple(replacement.shape)} does not match "
                f"expert output {tuple(output.shape)}."
            )
        return replacement

    return hook
