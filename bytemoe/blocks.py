from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SwiGLUWeights:
    """Row-major gate/up and column-major down projections for one expert."""

    gate: torch.Tensor  # [intermediate, hidden]
    up: torch.Tensor  # [intermediate, hidden]
    down: torch.Tensor  # [hidden, intermediate]

    @property
    def hidden_size(self) -> int:
        return self.gate.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.gate.shape[0]


def execute_expert(x: torch.Tensor, weights: SwiGLUWeights, indices: torch.Tensor | None = None) -> torch.Tensor:
    """Execute an exact SwiGLU expert, or the additive contribution of indices.

    `indices=None` executes all neurons. Summing disjoint calls over all indices
    recreates the full expert up to floating-point accumulation order.
    """
    if indices is None:
        gate, up, down = weights.gate, weights.up, weights.down
    else:
        gate = weights.gate.index_select(0, indices)
        up = weights.up.index_select(0, indices)
        down = weights.down.index_select(1, indices)
    activations = F.silu(x @ gate.T) * (x @ up.T)
    return activations @ down.T


def block_indices(order: torch.Tensor, num_blocks: int) -> list[torch.Tensor]:
    """Split an ordering into non-empty, near-equal additive neuron blocks."""
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    return [part for part in torch.tensor_split(order, num_blocks) if part.numel()]


def execute_blocks(x: torch.Tensor, weights: SwiGLUWeights, blocks: Iterable[torch.Tensor]) -> torch.Tensor:
    """Accumulate independently executable blocks without materializing all activations."""
    output = torch.zeros((*x.shape[:-1], weights.hidden_size), device=x.device, dtype=x.dtype)
    for indices in blocks:
        output = output + execute_expert(x, weights, indices.to(x.device))
    return output


def static_importance_order(weights: SwiGLUWeights) -> torch.Tensor:
    """Cheap data-free ordering used only for the feasibility gate.

    It favors neurons with large input and output projection norms. A learned
    ordering replaces this heuristic after E1 succeeds.
    """
    score = weights.down.float().norm(dim=0) * (
        weights.gate.float().norm(dim=1) + weights.up.float().norm(dim=1)
    )
    return torch.argsort(score, descending=True)


def random_order(size: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randperm(size, generator=generator, device=device)
