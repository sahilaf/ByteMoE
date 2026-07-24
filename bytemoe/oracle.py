from __future__ import annotations

import torch


def greedy_local_oracle(block_outputs: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose a token-specific additive block subset using complete block outputs.

    This is an oracle upper bound: a deployable policy cannot observe every
    block output before deciding what to transfer.
    """
    if block_outputs.ndim != 3:
        raise ValueError("block_outputs must have shape [tokens, blocks, hidden]")
    tokens, blocks, _ = block_outputs.shape
    if not 1 <= budget <= blocks:
        raise ValueError(f"budget must be in [1, {blocks}]")

    full = block_outputs.sum(dim=1)
    partial = torch.zeros_like(full)
    residual = full.clone()
    available = torch.ones((tokens, blocks), device=block_outputs.device, dtype=torch.bool)
    selections = []
    block_norm_sq = block_outputs.float().square().sum(dim=-1)
    rows = torch.arange(tokens, device=block_outputs.device)

    for _ in range(budget):
        # Maximize ||residual||² - ||residual - block||² for each token.
        improvement = 2 * (residual[:, None, :].float() * block_outputs.float()).sum(dim=-1) - block_norm_sq
        improvement.masked_fill_(~available, float("-inf"))
        chosen = improvement.argmax(dim=1)
        selected = block_outputs[rows, chosen]
        partial = partial + selected
        residual = residual - selected
        available[rows, chosen] = False
        selections.append(chosen)

    return partial, torch.stack(selections, dim=1)
