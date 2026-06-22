# Copyright (C) 2026 Tencent.
# Accuracy tests for CuTe DSL MoE vs C++ kernel.

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import math
import time

import pytest
import torch

import hpc
from utils import allclose, calculate_errors, errors_to_string

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def _is_sm100_plus():
    """Check if GPU is SM100+ (Blackwell) which supports CuTe DSL FP8 MMA."""
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


def _get_cute_fuse_moe_fn():
    """Get the CuTe DSL fuse_moe function if available."""
    try:
        from hpc.cute_fuse_moe import fuse_moe_pertensor_fp8 as cute_fn
        return cute_fn
    except ImportError:
        return None


def _run_cpp_kernel(
    x, gate_up_weight, down_weight, gate_up_scale, down_scale,
    act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_total,
    shared_output=None,
):
    """Run the C++ kernel directly via torch.ops."""
    return torch.ops.hpc.fuse_moe_pertensor_fp8(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale,
        shared_output, rank_ep, num_expert_total, True, None,
    )


def _run_cute_kernel(
    x, gate_up_weight, down_weight, gate_up_scale, down_scale,
    act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_total,
    shared_output=None,
):
    """Run the CuTe DSL kernel."""
    cute_fn = _get_cute_fuse_moe_fn()
    if cute_fn is None:
        pytest.skip("CuTe DSL not available")
    return cute_fn(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale,
        rank_ep, num_expert_total, True, shared_output,
    )


# =============================================================================
# Tests for non-GEMM components (work on any GPU)
# =============================================================================

@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("num_expert", [128])
def test_count_and_gather(num_seq, num_topk, hidden_size, num_expert):
    """Test count_and_gather against C++ kernel output."""
    from hpc.cute_fuse_moe import count_and_gather as cute_count_and_gather

    dtype = torch.float8_e4m3fn
    intermediate_size = 512

    topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)

    num_seq_per_group_avg = num_seq * num_topk // num_expert

    # Run C++ kernel
    cpp_result = torch.ops.hpc.count_and_gather(
        x, topk_ids, num_expert, 0, intermediate_size, num_seq_per_group_avg
    )

    # Run CuTe DSL (PyTorch-based) kernel
    cute_result = cute_count_and_gather(
        x, topk_ids, num_expert, 0, intermediate_size, num_seq_per_group_avg
    )

    # Compare seqlens (exact match expected)
    cpp_seqlens = cpp_result[3]
    cute_seqlens = cute_result[3]
    assert torch.equal(cpp_seqlens, cute_seqlens), (
        f"seqlens mismatch: max_diff={torch.max(torch.abs(cpp_seqlens - cute_seqlens))}"
    )

    # Compare cu_seqlens (exact match expected)
    cpp_cu_seqlens = cpp_result[4]
    cute_cu_seqlens = cute_result[4]
    assert torch.equal(cpp_cu_seqlens, cute_cu_seqlens), (
        f"cu_seqlens mismatch"
    )

    # Compare gate_up_input (FP8, may differ due to different gather order)
    cpp_input = cpp_result[0].to(torch.float32)
    cute_input = cute_result[0].to(torch.float32)
    # The gather order may differ between implementations, but the set of values
    # should be the same (just potentially in different order within each expert group)
    # We verify that the total sum is close
    assert abs(cpp_input.sum() - cute_input.sum()) < 1e-3, (
        f"gate_up_input sum mismatch: cpp={cpp_input.sum()}, cute={cute_input.sum()}"
    )

    print(f"count_and_gather: seqlens match OK, cu_seqlens match OK, input sum close OK")


@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("intermediate_size", [512])
def test_activation_quant(num_seq, intermediate_size):
    """Test activation+quant against naive PyTorch reference."""
    from hpc.cute_fuse_moe import act_mul_and_quant as cute_act_quant

    total_num_seq = num_seq
    gate_up_output = torch.randn(
        (total_num_seq, intermediate_size), dtype=torch.bfloat16, device="cuda"
    )
    act_and_mul_scale = torch.randn((1,), dtype=torch.float, device="cuda")

    # CuTe DSL version
    cute_out = cute_act_quant(
        gate_up_output, act_and_mul_scale, total_num_seq, intermediate_size, True
    )

    # Naive reference
    def silu(x):
        return x / (1 + (-x).exp())
    gate, up = torch.chunk(gate_up_output.float(), 2, dim=1)
    naive_out = (silu(gate).to(torch.bfloat16) * up.to(torch.bfloat16)).float() * act_and_mul_scale
    naive_out = naive_out.to(torch.float8_e4m3fn)

    # Compare
    cute_f32 = cute_out.to(torch.float32)
    naive_f32 = naive_out.to(torch.float32)
    errors = calculate_errors(naive_f32, cute_f32)
    print(errors_to_string(errors))
    assert allclose(naive_f32, cute_f32, rtol=0.08, atol=0.1), (
        f"activation_quant mismatch:\n{errors_to_string(errors)}"
    )


@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
def test_reduce(num_seq, num_topk, hidden_size):
    """Test reduce against naive PyTorch reference."""
    from hpc.cute_fuse_moe import reduce as cute_reduce

    total_num_seq = num_seq * num_topk
    x = torch.randn((total_num_seq, hidden_size), dtype=torch.bfloat16, device="cuda")
    topk_pos = torch.randint(0, total_num_seq, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk

    # CuTe DSL version
    cute_out = cute_reduce(x, topk_pos, topk_scale, None)

    # Naive reference
    naive_out = torch.zeros(num_seq, hidden_size, dtype=torch.float32, device="cuda")
    for k in range(num_topk):
        pos_k = topk_pos[:, k]
        scale_k = topk_scale[:, k].unsqueeze(1)
        x_k = x[pos_k.long()]
        naive_out += scale_k * x_k.float()
    naive_out = naive_out.to(torch.bfloat16)

    cute_f32 = cute_out.to(torch.float32)
    naive_f32 = naive_out.to(torch.float32)
    errors = calculate_errors(naive_f32, cute_f32)
    print(errors_to_string(errors))
    assert allclose(naive_f32, cute_f32, rtol=0.08, atol=0.1), (
        f"reduce mismatch:\n{errors_to_string(errors)}"
    )


# =============================================================================
# Full pipeline tests (require SM100+ for CuTe DSL GEMM)
# =============================================================================

@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("rank_ep", [0])
@pytest.mark.parametrize("size_ep", [1])
@pytest.mark.parametrize("has_shared_output", [False, True])
def test_cute_vs_cpp_accuracy(
    num_seq, num_topk, hidden_size, intermediate_size,
    num_expert, rank_ep, size_ep, has_shared_output,
):
    """Compare CuTe DSL output against C++ kernel output."""
    if _get_cute_fuse_moe_fn() is None:
        pytest.skip("CuTe DSL not available")

    dtype = torch.float8_e4m3fn
    num_expert_local = num_expert // size_ep

    topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    topk_ids, _ = torch.sort(topk_ids, dim=1)

    x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
    gate_up_weight = torch.randn(
        (num_expert_local, intermediate_size * 2, hidden_size),
        dtype=torch.float, device="cuda",
    ).to(dtype)
    down_weight = torch.randn(
        (num_expert_local, hidden_size, intermediate_size),
        dtype=torch.float, device="cuda",
    ).to(dtype)
    gate_up_scale = torch.randn((num_expert_local,), dtype=torch.float, device="cuda")
    down_scale = torch.randn((num_expert_local,), dtype=torch.float, device="cuda")
    act_and_mul_scale = torch.randn((1,), dtype=torch.float, device="cuda")
    topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk

    if has_shared_output:
        shared_output = torch.randn((num_seq, hidden_size), dtype=torch.bfloat16, device="cuda")
    else:
        shared_output = None

    # Run C++ kernel
    cpp_out = _run_cpp_kernel(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_local,
        shared_output,
    )

    # Run CuTe DSL kernel
    cute_out = _run_cute_kernel(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_local,
        shared_output,
    )

    # Compare
    cpp_f32 = cpp_out.to(torch.float32)
    cute_f32 = cute_out.to(torch.float32)

    errors = calculate_errors(cpp_f32, cute_f32)
    print(errors_to_string(errors))

    # Use same tolerances as existing tests
    assert allclose(cpp_f32, cute_f32, rtol=0.08, atol=0.1), (
        f"CuTe DSL output differs from C++ kernel output:\n{errors_to_string(errors)}"
    )


@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("rank_ep", [0])
@pytest.mark.parametrize("size_ep", [1])
@pytest.mark.parametrize("has_shared_output", [False])
def test_cute_vs_naive_accuracy(
    num_seq, num_topk, hidden_size, intermediate_size,
    num_expert, rank_ep, size_ep, has_shared_output,
):
    """Compare CuTe DSL output against naive PyTorch reference."""
    if _get_cute_fuse_moe_fn() is None:
        pytest.skip("CuTe DSL not available")

    dtype = torch.float8_e4m3fn
    num_expert_local = num_expert // size_ep

    topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    topk_ids, _ = torch.sort(topk_ids, dim=1)

    x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
    gate_up_weight = torch.randn(
        (num_expert_local, intermediate_size * 2, hidden_size),
        dtype=torch.float, device="cuda",
    ).to(dtype)
    down_weight = torch.randn(
        (num_expert_local, hidden_size, intermediate_size),
        dtype=torch.float, device="cuda",
    ).to(dtype)
    gate_up_scale = torch.randn((num_expert_local,), dtype=torch.float, device="cuda")
    down_scale = torch.randn((num_expert_local,), dtype=torch.float, device="cuda")
    act_and_mul_scale = torch.randn((1,), dtype=torch.float, device="cuda")
    topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk
    shared_output = None

    # Run CuTe DSL kernel
    cute_out = _run_cute_kernel(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_local,
        shared_output,
    )

    # Import naive reference from existing test
    from test_fuse_moe_pertensor import naive_fuse_moe_pertensor_fp8

    naive_out = naive_fuse_moe_pertensor_fp8(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale, rank_ep, shared_output,
    )

    cute_f32 = cute_out.to(torch.float32)
    naive_f32 = naive_out.to(torch.float32)

    errors = calculate_errors(naive_f32, cute_f32)
    print(errors_to_string(errors))

    assert allclose(naive_f32, cute_f32, rtol=0.08, atol=0.1), (
        f"CuTe DSL output differs from naive reference:\n{errors_to_string(errors)}"
    )


@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("rank_ep", [0])
@pytest.mark.parametrize("size_ep", [1])
def test_cute_vs_cpp_performance(
    num_seq, num_topk, hidden_size, intermediate_size,
    num_expert, rank_ep, size_ep,
):
    """Compare CuTe DSL vs C++ kernel performance."""
    if _get_cute_fuse_moe_fn() is None:
        pytest.skip("CuTe DSL not available")

    dtype = torch.float8_e4m3fn
    num_expert_local = num_expert // size_ep

    topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    topk_ids, _ = torch.sort(topk_ids, dim=1)

    x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
    gate_up_weight = torch.randn(
        (num_expert_local, intermediate_size * 2, hidden_size),
        dtype=torch.float, device="cuda",
    ).to(dtype)
    down_weight = torch.randn(
        (num_expert_local, hidden_size, intermediate_size),
        dtype=torch.float, device="cuda",
    ).to(dtype)
    gate_up_scale = torch.randn((num_expert_local,), dtype=torch.float, device="cuda")
    down_scale = torch.randn((num_expert_local,), dtype=torch.float, device="cuda")
    act_and_mul_scale = torch.randn((1,), dtype=torch.float, device="cuda")
    topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk
    shared_output = None

    warmup_iters = 5
    timed_iters = 10

    # Warmup C++ kernel
    for _ in range(warmup_iters):
        _run_cpp_kernel(
            x, gate_up_weight, down_weight, gate_up_scale, down_scale,
            act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_local,
            shared_output,
        )
    torch.cuda.synchronize()

    # Time C++ kernel
    cpp_times = []
    for _ in range(timed_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_cpp_kernel(
            x, gate_up_weight, down_weight, gate_up_scale, down_scale,
            act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_local,
            shared_output,
        )
        end.record()
        torch.cuda.synchronize()
        cpp_times.append(start.elapsed_time(end))

    # Warmup CuTe DSL kernel
    for _ in range(warmup_iters):
        _run_cute_kernel(
            x, gate_up_weight, down_weight, gate_up_scale, down_scale,
            act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_local,
            shared_output,
        )
    torch.cuda.synchronize()

    # Time CuTe DSL kernel
    cute_times = []
    for _ in range(timed_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_cute_kernel(
            x, gate_up_weight, down_weight, gate_up_scale, down_scale,
            act_and_mul_scale, topk_ids, topk_scale, rank_ep, num_expert_local,
            shared_output,
        )
        end.record()
        torch.cuda.synchronize()
        cute_times.append(start.elapsed_time(end))

    cpp_avg = sum(cpp_times) / len(cpp_times)
    cute_avg = sum(cute_times) / len(cute_times)

    print(f"\nPerformance comparison ({num_seq} tokens, {num_expert_local} experts):")
    print(f"  C++ kernel:     {cpp_avg:.4f} ms (avg of {timed_iters})")
    print(f"  CuTe DSL kernel: {cute_avg:.4f} ms (avg of {timed_iters})")
    print(f"  Ratio (CuTe/C++): {cute_avg / cpp_avg:.2f}x")

    # Don't assert on performance - just report
