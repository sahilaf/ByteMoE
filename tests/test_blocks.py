import torch

from bytemoe.blocks import SwiGLUWeights, block_indices, execute_blocks, execute_expert, static_importance_order


def test_all_blocks_reconstruct_expert_fp32():
    torch.manual_seed(7)
    weights = SwiGLUWeights(torch.randn(12, 5), torch.randn(12, 5), torch.randn(5, 12))
    x = torch.randn(4, 5)
    blocks = block_indices(static_importance_order(weights), 4)
    torch.testing.assert_close(execute_blocks(x, weights, blocks), execute_expert(x, weights), rtol=1e-5, atol=1e-5)
