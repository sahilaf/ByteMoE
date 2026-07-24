import torch

from bytemoe.oracle import greedy_local_oracle


def test_oracle_selects_unique_blocks_and_reconstructs_with_full_budget():
    blocks = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], [[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]]])
    partial, selections = greedy_local_oracle(blocks, budget=3)
    torch.testing.assert_close(partial, blocks.sum(dim=1))
    assert all(torch.unique(row).numel() == 3 for row in selections)
