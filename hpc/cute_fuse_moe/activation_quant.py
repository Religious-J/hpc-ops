# Copyright (C) 2026 Tencent.
# CuTe DSL MoE: SiLU activation + multiply + per-tensor FP8 quantization.
#
# This is a bandwidth-bound element-wise operation, implemented with PyTorch
# for efficiency. The operation is:
#   down_input = quantize_fp8(silu(gate) * up * act_and_mul_scale)
# where gate = gate_up_output[:, :intermediate//2], up = gate_up_output[:, intermediate//2:]

import torch
from torch import Tensor

_FP8_MAX = 448.0  # max magnitude for fp8_e4m3fn

# NOTE: This op is intentionally NOT torch.compile'd. Inductor lowers the
# silu/multiply chain slightly differently, which changes FP8 rounding at the
# quantization boundary enough to exceed the downstream accuracy tolerance.
# The eager version below matches the C++ kernel's numerics exactly.


def act_mul_and_quant(
    gate_up_output: Tensor,
    act_and_mul_scale: Tensor,
    total_num_seq: int,
    intermediate_size: int,
    use_bf16_mul: bool = True,
) -> Tensor:
    """
    Apply SiLU activation to the gate half, multiply with the up half,
    scale, and quantize to FP8.

    Args:
        gate_up_output: Gate+up GEMM output [total_num_seq, intermediate_size], dtype bf16
        act_and_mul_scale: Per-tensor scale factor, dtype float32
        total_num_seq: Total number of tokens (num_seq * num_topk)
        intermediate_size: Full intermediate size (gate + up, = 2 * actual_intermediate)
        use_bf16_mul: Whether to use bf16 for the multiply (True) or fp32 (False)

    Returns:
        down_input: Quantized FP8 tensor [total_num_seq, intermediate_size // 2]
    """
    half = intermediate_size // 2
    gate = gate_up_output[:, :half]
    up = gate_up_output[:, half:]

    gate_fp32 = gate.float()
    silu_gate = gate_fp32 * torch.sigmoid(gate_fp32)

    if use_bf16_mul:
        result = (silu_gate.to(torch.bfloat16) * up).float()
    else:
        result = silu_gate * up.float()

    result = result * act_and_mul_scale
    result = result.clamp(-_FP8_MAX, _FP8_MAX)
    return result.to(torch.float8_e4m3fn)
