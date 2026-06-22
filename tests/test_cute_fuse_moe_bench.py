# Copyright (C) 2026 Tencent.
# Benchmark: CuTe DSL MoE vs C++ kernel performance comparison.

"""
Compares CuTe DSL MoE implementation against the C++ kernel across
model presets and batch sizes.

Usage:
    python tests/test_cute_fuse_moe_bench.py
    python tests/test_cute_fuse_moe_bench.py --model deepseek-v3 --batch-sizes 4,16,64,256
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import torch
import hpc
from utils import calculate_errors, errors_to_string


# Model presets matching the existing benchmark
MODEL_PRESETS = {
    "qwen3-235b": {
        "num_expert": 128,
        "num_topk": 8,
        "hidden_size": 4096,
        "intermediate_size": 1536,
    },
    "hunyuan-v3": {
        "num_expert": 192,
        "num_topk": 8,
        "hidden_size": 4096,
        "intermediate_size": 1536,
    },
    "deepseek-v3": {
        "num_expert": 256,
        "num_topk": 8,
        "hidden_size": 7168,
        "intermediate_size": 2048,
    },
    "small": {
        "num_expert": 64,
        "num_topk": 8,
        "hidden_size": 2048,
        "intermediate_size": 512,
    },
}


def _get_cute_fn():
    try:
        from hpc.cute_fuse_moe import fuse_moe_pertensor_fp8 as cute_fn
        return cute_fn
    except ImportError:
        return None


def run_cpp_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                   act_and_mul_scale, topk_ids, topk_scale, num_expert_local, shared_output=None):
    return torch.ops.hpc.fuse_moe_pertensor_fp8(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale,
        shared_output, 0, num_expert_local, True, None,
    )


def run_cute_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                    act_and_mul_scale, topk_ids, topk_scale, num_expert_local, shared_output=None):
    cute_fn = _get_cute_fn()
    return cute_fn(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale,
        act_and_mul_scale, topk_ids, topk_scale,
        0, num_expert_local, True, shared_output,
    )


def benchmark_model(model_name, batch_sizes, warmup=5, timed_iters=10):
    """Benchmark both implementations for a given model preset."""
    if model_name not in MODEL_PRESETS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_PRESETS.keys())}")
        return

    preset = MODEL_PRESETS[model_name]
    num_expert = preset["num_expert"]
    num_topk = preset["num_topk"]
    hidden_size = preset["hidden_size"]
    intermediate_size = preset["intermediate_size"]
    dtype = torch.float8_e4m3fn

    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"  Experts={num_expert}, topk={num_topk}, hidden={hidden_size}, intermediate={intermediate_size}")
    print(f"{'='*80}")

    results = []

    for num_seq in batch_sizes:
        print(f"\n  Batch size: {num_seq}")

        # Generate inputs
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
        topk_ids, _ = torch.sort(topk_ids, dim=1)

        x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
        gate_up_weight = torch.randn(
            (num_expert, intermediate_size * 2, hidden_size),
            dtype=torch.float, device="cuda",
        ).to(dtype)
        down_weight = torch.randn(
            (num_expert, hidden_size, intermediate_size),
            dtype=torch.float, device="cuda",
        ).to(dtype)
        gate_up_scale = torch.randn((num_expert,), dtype=torch.float, device="cuda")
        down_scale = torch.randn((num_expert,), dtype=torch.float, device="cuda")
        act_and_mul_scale = torch.randn((1,), dtype=torch.float, device="cuda")
        topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk

        # --- C++ kernel ---
        # Warmup
        for _ in range(warmup):
            run_cpp_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                           act_and_mul_scale, topk_ids, topk_scale, num_expert)
        torch.cuda.synchronize()

        cpp_times = []
        for _ in range(timed_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_cpp_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                           act_and_mul_scale, topk_ids, topk_scale, num_expert)
            end.record()
            torch.cuda.synchronize()
            cpp_times.append(start.elapsed_time(end))

        cpp_avg = sum(cpp_times) / len(cpp_times)
        cpp_min = min(cpp_times)

        # --- CuTe DSL kernel ---
        cute_fn = _get_cute_fn()
        if cute_fn is not None:
            for _ in range(warmup):
                run_cute_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                                act_and_mul_scale, topk_ids, topk_scale, num_expert)
            torch.cuda.synchronize()

            cute_times = []
            for _ in range(timed_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_cute_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                                act_and_mul_scale, topk_ids, topk_scale, num_expert)
                end.record()
                torch.cuda.synchronize()
                cute_times.append(start.elapsed_time(end))

            cute_avg = sum(cute_times) / len(cute_times)
            cute_min = min(cute_times)
            ratio = cute_avg / cpp_avg
        else:
            cute_avg = float("nan")
            cute_min = float("nan")
            ratio = float("nan")

        # --- Accuracy comparison ---
        cpp_out = run_cpp_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                                 act_and_mul_scale, topk_ids, topk_scale, num_expert)
        if cute_fn is not None:
            cute_out = run_cute_kernel(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                                       act_and_mul_scale, topk_ids, topk_scale, num_expert)
            errors = calculate_errors(cpp_out.to(torch.float32), cute_out.to(torch.float32))
            max_abs = errors["top_abs_errors"][0]["error_value"] if errors["top_abs_errors"] else 0
            mean_abs = errors["mean_abs_error"]
        else:
            max_abs = float("nan")
            mean_abs = float("nan")

        result = {
            "model": model_name,
            "batch_size": num_seq,
            "cpp_avg_ms": cpp_avg,
            "cpp_min_ms": cpp_min,
            "cute_avg_ms": cute_avg,
            "cute_min_ms": cute_min,
            "ratio": ratio,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
        }
        results.append(result)

        print(f"    C++:     {cpp_avg:.4f} ms avg, {cpp_min:.4f} ms min")
        if cute_fn is not None:
            print(f"    CuTe DSL: {cute_avg:.4f} ms avg, {cute_min:.4f} ms min")
            print(f"    Ratio:    {ratio:.2f}x")
            print(f"    Accuracy: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CuTe DSL MoE vs C++ kernel")
    parser.add_argument("--model", type=str, default="small",
                        help=f"Model preset: {list(MODEL_PRESETS.keys())}")
    parser.add_argument("--batch-sizes", type=str, default="4,16,64,128,256",
                        help="Comma-separated batch sizes")
    parser.add_argument("--all-models", action="store_true",
                        help="Benchmark all model presets")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed iterations")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.all_models:
        models = list(MODEL_PRESETS.keys())
    else:
        models = [args.model]

    all_results = []
    for model in models:
        results = benchmark_model(model, batch_sizes, args.warmup, args.iters)
        all_results.extend(results)

    # Summary table
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    header = f"{'Model':<16} {'Batch':>8} {'C++(ms)':>10} {'CuTe(ms)':>10} {'Ratio':>8} {'MaxErr':>10} {'MeanErr':>10}"
    print(header)
    print("-" * 100)
    for r in all_results:
        print(f"{r['model']:<16} {r['batch_size']:>8} {r['cpp_avg_ms']:>10.4f} "
              f"{r['cute_avg_ms']:>10.4f} {r['ratio']:>8.2f} "
              f"{r['max_abs_error']:>10.6f} {r['mean_abs_error']:>10.6f}")


if __name__ == "__main__":
    main()
