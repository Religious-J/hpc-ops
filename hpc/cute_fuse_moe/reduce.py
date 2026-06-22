# Copyright (C) 2026 Tencent.
# CuTe DSL MoE: Scatter-add reduction with topk weights.
#
# This is a bandwidth-bound operation, implemented with PyTorch for efficiency.
# The operation is:
#   y[row] = sum_k(topk_scale[row, k] * x[topk_pos[row, k], :]) + shared_output[row, :]

import torch
from torch import Tensor
from typing import Optional


@torch.compile(dynamic=True)
def _reduce_core(
    x: Tensor,
    topk_pos: Tensor,
    topk_scale: Tensor,
    shared_output: Optional[Tensor],
) -> Tensor:
    num_seq, num_topk = topk_pos.shape
    hidden_size = x.size(1)
    gathered = x[topk_pos.reshape(-1).long()].view(num_seq, num_topk, hidden_size)
    # Weighted sum over topk as an elementwise multiply + reduction (not a matmul),
    # so it stays in true fp32 and is not lowered to a TF32 tensor-core path.
    y = (gathered.float() * topk_scale.unsqueeze(-1)).sum(dim=1)
    if shared_output is not None:
        y = y + shared_output.float()
    return y.to(torch.bfloat16)


def reduce(
    x: Tensor,
    topk_pos: Tensor,
    topk_scale: Tensor,
    shared_output: Optional[Tensor] = None,
) -> Tensor:
    """
    Weighted scatter-add reduction for MoE output.

    Aggregates weighted expert outputs back to original sequence positions.

    Args:
        x: Expert outputs [total_num_seq, hidden_size], dtype bf16
        topk_pos: Position indices [num_seq, num_topk], dtype int32
        topk_scale: Expert weights [num_seq, num_topk], dtype float32
        shared_output: Optional shared expert output [num_seq, hidden_size], dtype bf16

    Returns:
        y: Reduced output [num_seq, hidden_size], dtype bf16
    """
    # Vectorized gather + weighted topk-reduction, fused via torch.compile.
    return _reduce_core(x, topk_pos, topk_scale, shared_output)
