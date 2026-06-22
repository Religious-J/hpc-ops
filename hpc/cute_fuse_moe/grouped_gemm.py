# Copyright (C) 2026 Tencent.
#
# SM90 (Hopper) FP8 grouped GEMM for MoE, implemented with CuTe DSL.
#
# This is a CuTe DSL (Python) re-implementation of the C++/CUDA SM90 grouped GEMM
# kernel that previously lived in src/group_gemm/. It targets H20 / sm_90a.
#
# It computes, for each expert (group) g with token range
# [cu_seqlens[g], cu_seqlens[g+1]):
#       C[row, :] = (A[row, :] @ B[g].T) * scale[g]
# where
#   A : [total_tokens, K]   fp8_e4m3   (row-major, contiguous over all experts)
#   B : [N, K, num_experts] fp8_e4m3   (per-expert weight, K leading)
#   C : [total_tokens, N]   bf16        (row-major, contiguous over all experts)
#   scale : [num_experts]   float32
#
# Performance design (mirrors the C++ kernel):
#   The GEMM is computed *transposed* -- C^T = W @ A^T -- so that the (small)
#   per-expert token count lands on the MMA N axis instead of the M axis. This
#   uses a small-N WGMMA tile (SM90_64xNx32) and avoids the ~87% wasted compute
#   that a 64-row M tile would incur for MoE's ~8-token experts. The MMA A
#   operand is the weight (N_w rows), the B operand is the tokens; the
#   accumulator is C^T [N_w, tok] and is stored through a transposed view of C.
#
# Notes vs C++:
#   - The C++ kernel uses an SM90_64x8x32 MMA (token tile = 8). CuTe DSL's WGMMA
#     path requires N >= 16, so we use token tile = 16 (still far less waste than
#     a 64 M-tile). Everything else (TMA load, multi-stage pipeline) is mirrored.

import math
from typing import Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils


class GroupedGemmKernel:
    """SM90 FP8 grouped GEMM kernel (CuTe DSL), transposed small-tile design.

    :param acc_dtype: accumulator dtype (Float32)
    :param tile_shape_mn: logical CTA output tile (M=tokens, N=weight-N). The
        token dimension is clamped to the WGMMA minimum (16).
    """

    # WGMMA N axis (= token tile) minimum supported by the CuTe DSL path.
    MIN_TOK_TILE = 16

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        tile_shape_mn: Tuple[int, int],
    ):
        self.acc_dtype = acc_dtype
        # tile_tok: tokens on MMA-N ; tile_nw: weight-N chunk on MMA-M.
        self.tile_tok = max(tile_shape_mn[0], self.MIN_TOK_TILE)
        self.tile_nw = 64
        self.tile_k = 128  # 32 (mma K) * 4

        self.cluster_shape_mn = (1, 1)
        self.cta_layout_mnk = None
        self.tiled_mma = None

        self.occupancy = 1
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 128
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.sW_layout = None
        self.sA_layout = None
        self.buffer_align_bytes = 1024

    # ------------------------------------------------------------------
    # Static setup
    # ------------------------------------------------------------------
    def _setup_attributes(self):
        # MMA: M = tile_nw (weight-N chunk), N = tile_tok (tokens), K = 32.
        # A operand = weight, B operand = tokens, both K-major.
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.w_layout.sm90_mma_major_mode(),
            self.a_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            (1, 1, 1),
            tiler_mn=(self.tile_nw, self.tile_tok),
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))

        tile_mnk = (self.tile_nw, self.tile_tok, self.tile_k)
        self.ab_stage = self._compute_stages(
            tile_mnk, self.a_dtype, self.b_dtype, self.smem_capacity, self.occupancy
        )

        # smem layouts: W is MMA-A (Nw, K), tokens are MMA-B (tok, K).
        self.sW_layout = sm90_utils.make_smem_layout_a(
            self.w_layout, tile_mnk, self.b_dtype, self.ab_stage
        )
        self.sA_layout = sm90_utils.make_smem_layout_b(
            self.a_layout, tile_mnk, self.a_dtype, self.ab_stage
        )

    # ------------------------------------------------------------------
    # Host-side entry  (signature kept stable for fuse_moe_pertensor)
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (total_M, K)              tokens
        mB: cute.Tensor,  # (N, K, num_experts)       weights
        mC: cute.Tensor,  # (total_M, N)              output
        cu_seqlens: cute.Tensor,  # (num_experts + 1,) int32
        scale: cute.Tensor,  # (num_experts,) float32
        num_experts: cutlass.Constexpr[int],
        max_m_tiles: cutlass.Constexpr[int],
        nw_tiles: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        self.a_dtype = mA.element_type  # token dtype (MMA-B)
        self.b_dtype = mB.element_type  # weight dtype (MMA-A)
        self.c_dtype = mC.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(mA)  # tokens (tok, K)
        self.w_layout = utils.LayoutEnum.from_tensor(mB)  # weights (N, K, E)
        self.c_layout = utils.LayoutEnum.from_tensor(mC)
        self.num_experts = num_experts

        self._setup_attributes()

        # TMA: load weight tile (Nw, K) and token tile (tok, K).
        tma_atom_w, tma_w = self._make_tma_load(
            mB, self.sW_layout, (self.tile_nw, self.tile_k)
        )
        tma_atom_a, tma_a = self._make_tma_load(
            mA, self.sA_layout, (self.tile_tok, self.tile_k)
        )

        self.nw_tiles = nw_tiles

        # Static grid: (weight-N tiles, token-M tiles, experts). Out-of-range
        # token tiles for an expert exit early in the kernel.
        grid = (nw_tiles, max_m_tiles, num_experts)

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.sW_layout)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.sA_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_w,
            tma_w,
            tma_atom_a,
            tma_a,
            mC,
            self.tiled_mma,
            self.sW_layout,
            self.sA_layout,
            cu_seqlens,
            scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )
        return

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mW_nke: cute.Tensor,  # (N, K, num_experts)
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,  # (total_M, K)
        mC_mn: cute.Tensor,  # (total_M, N)
        tiled_mma: cute.TiledMma,
        sW_layout: cute.ComposedLayout,
        sA_layout: cute.ComposedLayout,
        cu_seqlens: cute.Tensor,
        scale: cute.Tensor,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)

        tile_nw, tile_tok, expert = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        m_start = cu_seqlens[expert]
        seqlen = cu_seqlens[expert + 1] - m_start

        # Skip token tiles past this expert's tokens (no early return in DSL).
        if tile_tok * self.tile_tok < seqlen:
            self._compute_tile(
                tma_atom_w,
                mW_nke,
                tma_atom_a,
                mA_mk,
                mC_mn,
                tiled_mma,
                sW_layout,
                sA_layout,
                scale,
                warp_idx,
                tidx,
                tile_nw,
                tile_tok,
                expert,
                m_start,
                seqlen,
            )
        return

    @cute.jit
    def _compute_tile(
        self,
        tma_atom_w,
        mW_nke,
        tma_atom_a,
        mA_mk,
        mC_mn,
        tiled_mma,
        sW_layout,
        sA_layout,
        scale,
        warp_idx,
        tidx,
        tile_nw,
        tile_tok,
        expert,
        m_start,
        seqlen,
    ):
        expert_scale = scale[expert]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sW = storage.sW.get_tensor(sW_layout.outer, swizzle=sW_layout.inner)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)

        sW1 = cute.slice_(sW_layout, (None, None, 0))
        sA1 = cute.slice_(sA_layout, (None, None, 0))
        tx_bytes = cute.size_in_bytes(self.b_dtype, sW1) + cute.size_in_bytes(
            self.a_dtype, sA1
        )

        mbar_ptr = storage.mbar.data_ptr()
        prod = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        nwarps = self.threads_per_cta // 32
        cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, nwarps)
        cta_layout_vmnk = cute.make_layout(
            (1, self.cluster_shape_mn[0], self.cluster_shape_mn[1], 1)
        )
        mp = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=prod,
            consumer_group=cons,
            tx_count=tx_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Per-expert token slice and weight (expert-indexed).
        mA_expert = cute.domain_offset((m_start, 0), mA_mk)  # (tok, K)
        mW_expert = mW_nke[(None, None, expert)]  # (N, K)

        tile_mnk = (self.tile_nw, self.tile_tok, self.tile_k)
        gW = cute.local_tile(
            mW_expert, tile_mnk, (tile_nw, tile_tok, None), proj=(1, None, 1)
        )  # (bNw, bK, restK)
        gA = cute.local_tile(
            mA_expert, tile_mnk, (tile_nw, tile_tok, None), proj=(None, 1, 1)
        )  # (btok, bK, restK)

        thr_mma = tiled_mma.get_slice(0)

        sW_tma = cute.group_modes(sW, 0, 2)
        gW_tma = cute.group_modes(gW, 0, 2)
        tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w, 0, cute.make_layout(1), sW_tma, gW_tma
        )
        sA_tma = cute.group_modes(sA, 0, 2)
        gA_tma = cute.group_modes(gA, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sA_tma, gA_tma
        )

        tCsW = thr_mma.partition_A(sW)
        tCsA = thr_mma.partition_B(sA)
        tCrW = tiled_mma.make_fragment_A(tCsW)
        tCrA = tiled_mma.make_fragment_B(tCsA)

        # accumulator C^T: (Nw, tok)
        acc_shape = thr_mma.partition_C(
            cute.make_identity_tensor((self.tile_nw, self.tile_tok))
        ).shape
        acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        k_tiles = cute.size(gW, mode=[2])
        prefetch = cutlass.max(cutlass.min(self.ab_stage, k_tiles), 0)

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if warp_idx == 0:
            for _ in cutlass.range(prefetch, unroll=1):
                mp.producer_acquire(prod_state)
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, prod_state.count)],
                    tWsW[(None, prod_state.index)],
                    tma_bar_ptr=mp.producer_get_barrier(prod_state),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, prod_state.count)],
                    tAsA[(None, prod_state.index)],
                    tma_bar_ptr=mp.producer_get_barrier(prod_state),
                    mcast_mask=0,
                )
                mp.producer_commit(prod_state)
                prod_state.advance()

        kpm = 1
        read = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        rel = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        peek = cutlass.Boolean(1)
        if read.count < k_tiles:
            peek = mp.consumer_try_wait(read)

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        nkb = cute.size(tCrW, mode=[2])

        for _ in cutlass.range_constexpr(kpm):
            mp.consumer_wait(read, peek)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range(nkb, unroll_full=True):
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrW[(None, None, kb, read.index)],
                    tCrA[(None, None, kb, read.index)],
                    acc,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            read.advance()
            peek = cutlass.Boolean(1)
            if read.count < k_tiles:
                peek = mp.consumer_try_wait(read)

        for k in cutlass.range(kpm, k_tiles, 1, unroll=1):
            mp.consumer_wait(read, peek)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range(nkb, unroll_full=True):
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrW[(None, None, kb, read.index)],
                    tCrA[(None, None, kb, read.index)],
                    acc,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(kpm)
            mp.consumer_release(rel)
            read.advance()
            rel.advance()
            peek = cutlass.Boolean(1)
            if read.count < k_tiles:
                peek = mp.consumer_try_wait(read)

        cute.nvgpu.warpgroup.wait_group(0)
        for _ in cutlass.range(kpm, unroll=1):
            mp.consumer_release(rel)
            rel.advance()
        cute.arch.sync_threads()

        # Epilogue: acc is C^T (Nw, tok). Apply per-expert scale, convert to
        # bf16, and store through a transposed (Nw, tok) view of C so each
        # element lands at C[token, weight_N]. Predicated on this expert's tokens.
        acc_vec = acc.load()
        acc_out = cute.make_rmem_tensor(acc_shape, self.c_dtype)
        acc_out.store((acc_vec * expert_scale).to(self.c_dtype))

        mC_expert = cute.domain_offset((m_start, 0), mC_mn)  # (tok, N)
        # Transposed view: mCt[nw, tok] aliases mC_expert[tok, nw].
        mCt = cute.make_tensor(
            mC_expert.iterator,
            cute.make_layout(
                (mC_expert.shape[1], mC_expert.shape[0]),
                stride=(mC_expert.stride[1], mC_expert.stride[0]),
            ),
        )
        gCt = cute.local_tile(mCt, (self.tile_nw, self.tile_tok), (tile_nw, tile_tok))

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tiled_copy_C = cute.make_tiled_copy_C(copy_atom, tiled_mma)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        tCrC = thr_copy_C.retile(acc_out)
        tCgCt = thr_copy_C.partition_D(gCt)

        # Row(token)-bound predicate. coords are (nw, tok) over the tile.
        cT = cute.make_identity_tensor((self.tile_nw, self.tile_tok))
        tCcT = thr_copy_C.partition_D(cT)
        pred = cute.make_rmem_tensor(tCgCt.shape, cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(pred)):
            tk = tCcT[i][1]
            pred[i] = (tile_tok * self.tile_tok + tk) < seqlen

        cute.copy(copy_atom, tCrC, tCgCt, pred=pred)
        return

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_stages(tile_mnk, a_dtype, b_dtype, smem_capacity, occupancy):
        a_shape = cute.slice_(tile_mnk, (None, 0, None))  # (Nw, K)
        b_shape = cute.slice_(tile_mnk, (0, None, None))  # (tok, K)
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        # Reserve some smem for the epilogue / alignment headroom.
        epi_bytes = 8192
        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return max(min(ab_stage, 8), 2)

    @staticmethod
    def _make_tma_load(tensor, smem_layout_staged, smem_tile):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=1,
        )
        return tma_atom, tma_tensor
