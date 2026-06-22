# Copyright (C) 2026 Tencent.
# CuTe DSL MoE: Count-and-gather using PyTorch operations.
#
# The count-and-gather step is bandwidth-bound, so we use efficient PyTorch
# operations. The gather order matches the C++ kernel: tokens are iterated in
# sequence order (row-major over [num_seq, num_topk]) and assigned to per-expert
# positions using running counters. This is equivalent to a stable sort by expert ID.

import torch
from torch import Tensor
from typing import Tuple


def count_and_gather(
    x: Tensor,
    topk_ids: Tensor,
    num_expert: int,
    rank_ep: int,
    intermediate_size: int,
    num_seq_per_group_avg: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Count tokens per expert and gather inputs into per-expert contiguous buffers.

    Args:
        x: Input token features [num_seq, hidden_size], dtype fp8
        topk_ids: Expert assignments [num_seq, num_topk], dtype int32
        num_expert: Number of experts on this device
        rank_ep: Expert parallelism rank
        intermediate_size: Intermediate (gate+up) dimension
        num_seq_per_group_avg: Average tokens per expert (for buffer sizing)

    Returns:
        gate_up_input: Gathered input [total_tokens, hidden_size], dtype fp8
        gate_up_output: Pre-allocated output buffer [total_tokens, intermediate_size], dtype bf16
        topk_pos: Position indices [num_seq, num_topk], dtype int32
        seqlens: Token counts per expert [num_expert], dtype int32
        cu_seqlens: Cumulative token counts [num_expert + 1], dtype int32
        tiles: Tile counts per expert [num_expert], dtype int32
        cu_tiles: Cumulative tile counts [num_expert + 1], dtype int32
        gate_up_tmas: TMA descriptors placeholder [num_expert * 2, 128], dtype int8
        down_tmas: TMA descriptors placeholder [num_expert * 2, 128], dtype int8
    """
    device = x.device
    num_seq = x.size(0)
    hidden_size = x.size(1)
    num_topk = topk_ids.size(1)
    total_tokens = num_seq * num_topk

    # --- Step 1: Count tokens per expert ---
    flat_ids = topk_ids.contiguous().view(-1).long()  # [num_seq * num_topk]
    seqlens = torch.bincount(flat_ids, minlength=num_expert).to(torch.int32)

    # --- Step 2: Cumulative seqlens ---
    cu_seqlens = torch.zeros(num_expert + 1, dtype=torch.int32, device=device)
    torch.cumsum(seqlens, 0, out=cu_seqlens[1:])

    # --- Step 3: topk_pos + gather (stable sort by expert id) ---
    # Stable sort groups tokens by expert while preserving row-major order
    # within each group. sorted_indices[i] = flat token index at sorted slot i.
    sorted_indices = torch.argsort(flat_ids, stable=True)

    # topk_pos[seq, topk] = sorted slot of token (seq, topk), so the downstream
    # reduce can scatter expert outputs back to their source rows.
    positions = torch.empty(total_tokens, dtype=torch.int32, device=device)
    positions[sorted_indices] = torch.arange(
        total_tokens, dtype=torch.int32, device=device
    )
    topk_pos = positions.view(num_seq, num_topk)

    # Gathered input: sorted slot i pulls hidden features from x[seq], where
    # seq = (flat token index) // num_topk.
    src_seq = sorted_indices // num_topk
    gate_up_input = x[src_seq].contiguous()

    # --- Step 4: Output buffer for the gate-up GEMM ---
    gate_up_output = torch.empty(
        total_tokens, intermediate_size, dtype=torch.bfloat16, device=device
    )

    # --- Step 5: Tile counts (fully vectorized, no host sync) ---
    _ALIGN_TABLE = (
        (8, 8),
        (16, 16),
        (32, 32),
        (48, 48),
        (64, 64),
        (96, 48),
        (128, 32),
        (144, 48),
    )
    aligned_size = 64
    for thr, val in _ALIGN_TABLE:
        if num_seq_per_group_avg <= thr:
            aligned_size = val
            break

    kTileN = 128
    num_n_tiles_gateup = (intermediate_size + kTileN - 1) // kTileN
    num_m_tiles = (seqlens + (aligned_size - 1)) // aligned_size
    tiles = (num_m_tiles * num_n_tiles_gateup).to(torch.int32)
    cu_tiles = torch.zeros(num_expert + 1, dtype=torch.int32, device=device)
    torch.cumsum(tiles, 0, out=cu_tiles[1:])

    # TMA descriptor placeholders kept for return-signature compatibility.
    gate_up_tmas = torch.empty(num_expert * 2, 128, dtype=torch.int8, device=device)
    down_tmas = gate_up_tmas

    return (
        gate_up_input,
        gate_up_output,
        topk_pos,
        seqlens,
        cu_seqlens,
        tiles,
        cu_tiles,
        gate_up_tmas,
        down_tmas,
    )
    topk_pos = positions.view(num_seq, num_topk)

    # Gather: token t = (seq, topk) maps to original row seq = t // num_topk.
    # The expert-sorted buffer's i-th row is the token at sorted_indices[i].
    # Each gathered row pulls hidden features from x[seq], where seq is derived
    # from the flat token index. (topk_pos[seq,topk] already records where each
    # (seq,topk) lands, so reduce can scatter results back.)
    src_seq = (sorted_indices // num_topk).long()  # original row per sorted slot
    gate_up_input = x[src_seq].contiguous()

    # --- Step 4: Allocate output buffers ---
    gate_up_output = torch.empty(
        total_tokens, intermediate_size, dtype=torch.bfloat16, device=device
    )
    down_input = torch.empty(
        total_tokens, intermediate_size // 2, dtype=torch.float8_e4m3fn, device=device
    )
    down_output = torch.empty(
        total_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    # --- Step 5: Compute tile counts ---
    if num_seq_per_group_avg <= 8:
        aligned_size = 8
    elif num_seq_per_group_avg <= 16:
        aligned_size = 16
    elif num_seq_per_group_avg <= 32:
        aligned_size = 32
    elif num_seq_per_group_avg <= 48:
        aligned_size = 48
    elif num_seq_per_group_avg <= 64:
        aligned_size = 64
    elif num_seq_per_group_avg <= 96:
        aligned_size = 48
    elif num_seq_per_group_avg <= 128:
        aligned_size = 32
    elif num_seq_per_group_avg <= 144:
        aligned_size = 48
    else:
        aligned_size = 64

    kTileN = 128
    tiles = torch.zeros(num_expert, dtype=torch.int32, device=device)
    for e in range(num_expert):
        num_tokens_e = seqlens[e].item()
        num_m_tiles = (num_tokens_e + aligned_size - 1) // aligned_size
        num_n_tiles_gateup = (intermediate_size + kTileN - 1) // kTileN
        tiles[e] = num_m_tiles * num_n_tiles_gateup

    cu_tiles = torch.zeros(num_expert + 1, dtype=torch.int32, device=device)
    cu_tiles[1:] = tiles.cumsum(0)

    # --- Step 6: TMA descriptors (placeholder) ---
    gate_up_tmas = torch.empty(num_expert * 2, 128, dtype=torch.int8, device=device)
    down_tmas = torch.empty(num_expert * 2, 128, dtype=torch.int8, device=device)

    return (
        gate_up_input,
        gate_up_output,
        topk_pos,
        seqlens,
        cu_seqlens,
        tiles,
        cu_tiles,
        gate_up_tmas,
        down_tmas,
    )
