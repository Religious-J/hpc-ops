# Copyright (C) 2026 Tencent.
# CuTe DSL MoE: Per-tensor FP8 fused MoE forward pass (SM90 / Hopper, e.g. H20).
#
# Pipeline: count+gather -> gate-up GEMM -> SiLU+multiply+quant -> down GEMM -> reduce
#
# The two grouped GEMMs run on a pure CuTe DSL SM90 FP8 kernel
# (hpc/cute_fuse_moe/grouped_gemm.py), a Python re-implementation of the former
# C++/CUDA src/group_gemm kernel.

import torch
from torch import Tensor
from typing import Optional, Tuple

from .activation_quant import act_mul_and_quant as _act_mul_and_quant
from .reduce import reduce as _reduce


def _count_and_gather_min(x, topk_ids, num_expert, intermediate_size):
    """Minimal count + gather: only the tensors the CuTe pipeline consumes.

    Returns (gate_up_input, gate_up_output, topk_pos, cu_seqlens). Avoids the
    tile-count / TMA-descriptor bookkeeping (unused by this pipeline) and the
    extra buffer allocations, which were a large part of the host-side overhead.
    """
    device = x.device
    num_seq = x.size(0)
    num_topk = topk_ids.size(1)
    total_tokens = num_seq * num_topk

    flat_ids = topk_ids.reshape(-1).long()
    seqlens = torch.bincount(flat_ids, minlength=num_expert).to(torch.int32)
    cu_seqlens = torch.zeros(num_expert + 1, dtype=torch.int32, device=device)
    torch.cumsum(seqlens, 0, out=cu_seqlens[1:])

    sorted_indices = torch.argsort(flat_ids, stable=True)
    positions = torch.empty(total_tokens, dtype=torch.int32, device=device)
    positions.scatter_(
        0,
        sorted_indices,
        torch.arange(total_tokens, dtype=torch.int32, device=device),
    )
    topk_pos = positions.view(num_seq, num_topk)

    src_seq = sorted_indices // num_topk
    gate_up_input = x[src_seq].contiguous()
    gate_up_output = torch.empty(
        total_tokens, intermediate_size, dtype=torch.bfloat16, device=device
    )
    return gate_up_input, gate_up_output, topk_pos, cu_seqlens


# Cache of compiled CuTe kernels keyed by static problem shape, so we compile
# once per (tile, N, K, num_experts) combination instead of every call.
_COMPILED_GG_CACHE = {}


def _run_grouped_gemm(
    mat_a: Tensor,  # [total_M, K]            fp8_e4m3fn
    weight: Tensor,  # [num_experts, N, K]     fp8_e4m3fn (K-contiguous)
    out: Tensor,  # [total_M, N]            bf16
    cu_seqlens: Tensor,  # [num_experts + 1]       int32
    scale: Tensor,  # [num_experts]           float32
    tile_mn: Tuple[int, int] = (64, 128),
) -> None:
    """Run an FP8 grouped GEMM on SM90 via the CuTe DSL kernel.

    Computes, per expert e over its token range [cu_seqlens[e], cu_seqlens[e+1]):
        out[row, :] = (mat_a[row, :] @ weight[e].T) * scale[e]
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from .grouped_gemm import GroupedGemmKernel

    num_experts = weight.size(0)
    N = weight.size(1)
    K = weight.size(2)

    # Transposed GEMM: token count is on the MMA N axis (tile_tok), weight-N on
    # the M axis. Static grid = (nw_tiles, max_m_tiles, experts); out-of-range
    # token tiles exit early in the kernel.
    tile_tok = max(tile_mn[0], 16)  # WGMMA N (token) minimum is 16
    total_m = mat_a.size(0)
    # Worst case: all tokens on a single expert -> bound token tiles by total.
    max_m_tiles = max((total_m + tile_tok - 1) // tile_tok, 1)

    # weight [E, N, K] (K-contiguous) -> logical (N, K, E), keeping K stride-1.
    weight_view = weight.as_strided((N, K, num_experts), (K, 1, N * K))

    # Zero-copy: bind cute tensors directly to the real torch storage via dlpack.
    # No per-call device allocation or host/device copies.
    a_cute = from_dlpack(mat_a, assumed_align=16)
    a_cute.element_type = cutlass.Float8E4M3FN
    a_cute = a_cute.mark_layout_dynamic(leading_dim=1)

    b_cute = from_dlpack(weight_view, assumed_align=16)
    b_cute.element_type = cutlass.Float8E4M3FN
    b_cute = b_cute.mark_layout_dynamic(leading_dim=1)  # K is leading

    c_cute = from_dlpack(out, assumed_align=16)
    c_cute.element_type = cutlass.BFloat16
    c_cute = c_cute.mark_layout_dynamic(leading_dim=1)

    cu_cute = from_dlpack(cu_seqlens, assumed_align=16)
    cu_cute.element_type = cutlass.Int32
    cu_cute = cu_cute.mark_layout_dynamic(leading_dim=0)

    sc_cute = from_dlpack(scale, assumed_align=16)
    sc_cute.element_type = cutlass.Float32
    sc_cute = sc_cute.mark_layout_dynamic(leading_dim=0)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    nw_tiles = (N + 64 - 1) // 64  # tile_nw = 64
    cache_key = (tile_mn, N, K, num_experts, max_m_tiles)
    compiled = _COMPILED_GG_CACHE.get(cache_key)
    if compiled is None:
        kernel = GroupedGemmKernel(acc_dtype=cutlass.Float32, tile_shape_mn=tile_mn)
        compiled = cute.compile(
            kernel,
            a_cute,
            b_cute,
            c_cute,
            cu_cute,
            sc_cute,
            num_experts,
            max_m_tiles,
            nw_tiles,
            stream,
        )
        _COMPILED_GG_CACHE[cache_key] = compiled

    compiled(a_cute, b_cute, c_cute, cu_cute, sc_cute, stream)


def fuse_moe_pertensor_fp8(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
) -> Tensor:
    """
    Run per-tensor FP8 FusedMoE on SM90 (Hopper / H20) with CuTe DSL GEMMs.

    Pipeline:
    1. Count tokens per expert and gather inputs (C++ count_and_gather op)
    2. Gate-up grouped GEMM (FP8 x FP8 -> BF16), scaled by gate_up_scale
    3. SiLU activation + multiply + per-tensor FP8 quantize
    4. Down grouped GEMM (FP8 x FP8 -> BF16), scaled by down_scale
    5. Scatter-add reduction with topk weights

    Args:
        x: Input tokens [num_seq, hidden_size], dtype fp8_e4m3fn
        gate_up_weight: Gate+up weight [num_expert_local, intermediate_size, hidden_size], dtype fp8_e4m3fn
        down_weight: Down weight [num_expert_local, hidden_size, intermediate_size//2], dtype fp8_e4m3fn
        gate_up_scale: Gate-up weight scale [num_expert_local], dtype float32
        down_scale: Down weight scale [num_expert_local], dtype float32
        act_and_mul_scale: Activation scale, dtype float32
        topk_ids: Expert assignments [num_seq, num_topk], dtype int32
        topk_scale: Expert weights [num_seq, num_topk], dtype float32
        rank_ep: Expert parallelism rank
        num_expert_total: Total number of experts across all ranks
        use_bf16_mul: Use bf16 for activation multiply
        shared_output: Optional shared expert output [num_seq, hidden_size], dtype bf16
        output: Optional pre-allocated output tensor

    Returns:
        y: MoE output [num_seq, hidden_size], dtype bf16
    """
    num_seq = x.size(0)
    hidden_size = x.size(1)
    num_expert_local = gate_up_weight.size(0)
    intermediate_size = gate_up_weight.size(1)  # gate+up fused: 2 * actual_intermediate
    num_topk = topk_ids.size(1)
    total_num_seq = num_seq * num_topk
    num_seq_per_group_avg = total_num_seq // num_expert_total

    # --- Step 1: Count and gather (pure-Python; no C++ dependency) ---
    gate_up_input, gate_up_output, topk_pos, cu_seqlens = _count_and_gather_min(
        x, topk_ids, num_expert_local, intermediate_size
    )

    # --- Step 2: Gate-up grouped GEMM ---
    # gate_up_input: [total_tokens, hidden_size] FP8
    # gate_up_weight: [num_expert, intermediate_size, hidden_size] FP8 (K=hidden contiguous)
    _run_grouped_gemm(
        mat_a=gate_up_input,
        weight=gate_up_weight,
        out=gate_up_output,
        cu_seqlens=cu_seqlens,
        scale=gate_up_scale,
    )

    # --- Step 3: SiLU + multiply + quantize ---
    down_input = _act_mul_and_quant(
        gate_up_output,
        act_and_mul_scale,
        total_num_seq,
        intermediate_size,
        use_bf16_mul,
    )

    # --- Step 4: Down grouped GEMM ---
    # down_input: [total_tokens, intermediate_size//2] FP8
    # down_weight: [num_expert, hidden_size, intermediate_size//2] FP8 (K=intermediate//2 contiguous)
    down_output = torch.empty(
        total_num_seq, hidden_size, dtype=torch.bfloat16, device=x.device
    )

    _run_grouped_gemm(
        mat_a=down_input,
        weight=down_weight,
        out=down_output,
        cu_seqlens=cu_seqlens,
        scale=down_scale,
    )

    # --- Step 5: Reduce ---
    y = _reduce(down_output, topk_pos, topk_scale, shared_output)

    if output is not None:
        output.copy_(y)
        return output

    return y
