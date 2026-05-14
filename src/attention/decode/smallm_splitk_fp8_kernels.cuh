// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_DECODE_SMALLM_SPLITK_FP8_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SMALLM_SPLITK_FP8_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/decode/util_kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

// Swizzle safety (must stay in sync with SLayoutQ / SLayoutY / SLayoutSplitY):
// - Q load: num_dim_qk multiple of 8; uint4 never crosses head boundaries.
// - Y store: Swizzle period 16 BF16; d aligned to 8 => uint4 safe; float->BF16 in registers.
// - splitY store: Swizzle period 8 floats; d aligned to 4 => float4 safe.

// kTileK: compile-time head dimension (= num_dim_qk, always 128 in this kernel).
// Passing it as a template int lets the compiler replace runtime div/mod with
// cheap shift-and-mask (e.g. / 128 => >> 7, % 128 => & 127).
template <typename Tin, int kTileK, typename TensorQG, typename TensorSQ>
__device__ __forceinline__ void load_q_group_direct_to_smem_fp8(
    TensorQG const &Q, TensorSQ &sQ, int ihead_kv, int ibatch,
    int heads_per_group, int num_seq_q, int pad_heads_per_group,
    int rank_in_threads, int num_threads) {
  using namespace cute;  // NOLINT

  constexpr int kVecSize  = 16;  // uint4 = 128 bits = 16 FP8 elements
  constexpr int kVecsPerHead = kTileK / kVecSize;
  const int total_elems   = heads_per_group * kTileK;
  const int kVecStride    = num_threads * kVecSize;

  const int rank_vec = rank_in_threads;
  const int lh0 = rank_vec / kVecsPerHead;
  const int k0  = (rank_vec % kVecsPerHead) * kVecSize;

  for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
    // Q layout: (heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch)
    const Tin *q_base = Q(_, 0, ihead_kv, iseqq, ibatch).data().get();
    int lh = lh0, k = k0;
    for (int base = rank_in_threads * kVecSize; base + kVecSize <= total_elems;
         base += kVecStride) {
      int srow = iseqq * pad_heads_per_group + lh;
      store(&sQ(srow, k), load<Tin, kVecSize>(q_base + base));
      k += kVecStride % kTileK;
      if (k >= kTileK) { k -= kTileK; lh++; }
      lh += kVecStride / kTileK;
    }
    // Scalar tail
    const int vec_covered = (total_elems / kVecStride) * kVecStride;
    for (int elem = vec_covered + rank_in_threads; elem < total_elems; elem += num_threads) {
      int elh  = elem / kTileK;
      int ek   = elem % kTileK;
      int srow = iseqq * pad_heads_per_group + elh;
      sQ(srow, ek) = Q(elh, ek, ihead_kv, iseqq, ibatch);
    }
  }
}

// kTileV: compile-time V head-dimension (always 128).
template <typename Tout, int kTileV, typename TensorSY, typename TensorGY>
__device__ __forceinline__ void store_sY_to_gmem_bf16_fp8(
    TensorSY &sY, TensorGY &Y, int ihead_kv, int ibatch, int heads_per_group,
    int num_seq_q, int pad_heads_per_group, int iwarpgroup,
    int idx, int kMathThreads) {
  using namespace cute;  // NOLINT

  constexpr int vec_size      = 8;
  constexpr int num_dim_v_vec = kTileV / vec_size;
  static_assert(kTileV % vec_size == 0, "kTileV must be a multiple of 8");

  const int total_vec = heads_per_group * num_dim_v_vec;

  const int lh0 = idx / num_dim_v_vec;
  const int d0  = (idx % num_dim_v_vec) * vec_size;

  for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
    int lh = lh0, d = d0;
    for (int lin = idx; lin < total_vec; lin += kMathThreads) {
      int sY_lh = iseqq * pad_heads_per_group + lh;
      store(&Y(d, lh, ihead_kv, iseqq, ibatch),
            load<Tout, vec_size>(&sY(d, sY_lh, iwarpgroup)));
      lh += kMathThreads / num_dim_v_vec;
      d  += (kMathThreads % num_dim_v_vec) * vec_size;
      if (d >= kTileV) { d -= kTileV; lh++; }
    }
  }
}

template <typename TensorSSplitY, typename TensorGSplitY>
__device__ __forceinline__ void store_sSplitY_to_gmem_float_fp8(
    TensorSSplitY &sSplitY, TensorGSplitY &splitY, int ihead_kv, int ichunk, int ibatch,
    int heads_per_group, int num_seq_q, int pad_heads_per_group, int num_dim_v,
    int iwarpgroup, int idx, int kMathThreads) {
  using namespace cute;  // NOLINT

  const int vec_size      = 4;
  const int num_dim_v_vec = num_dim_v / vec_size;
  const int num_dim_v_rem = num_dim_v % vec_size;
  const int total_vec     = heads_per_group * num_dim_v_vec;
  for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
    int sY_off = iseqq * pad_heads_per_group;
    for (int lin = idx; lin < total_vec; lin += kMathThreads) {
      int lh    = lin / num_dim_v_vec;
      int d_idx = lin % num_dim_v_vec;
      int d     = d_idx * vec_size;
      int sY_lh = sY_off + lh;
      float4 val;
      val.x = sSplitY(d, sY_lh, iwarpgroup);
      val.y = sSplitY(d + 1, sY_lh, iwarpgroup);
      val.z = sSplitY(d + 2, sY_lh, iwarpgroup);
      val.w = sSplitY(d + 3, sY_lh, iwarpgroup);
      store(&splitY(d, lh, ihead_kv, iseqq, ichunk, ibatch), load<float, vec_size>(&val));
    }
    if (num_dim_v_rem > 0) {
      const int d_base    = num_dim_v_vec * vec_size;
      const int total_rem = heads_per_group * num_dim_v_rem;
      for (int lin = idx; lin < total_rem; lin += kMathThreads) {
        int lh    = lin / num_dim_v_rem;
        int d     = d_base + lin % num_dim_v_rem;
        int sY_lh = sY_off + lh;
        splitY(d, lh, ihead_kv, iseqq, ichunk, ibatch) = sSplitY(d, sY_lh, iwarpgroup);
      }
    }
  }
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          int kHeadsPerGroup, int kWarpGroupN, typename TiledMmaQK, typename TiledMmaSV,
          typename TmaQ, typename TmaK, typename TmaV, typename TmaY, typename TmaSplitY,
          typename TensorQ, typename TensorY, typename TensorSplitY,
          typename SLayoutQ, typename SLayoutK, typename SLayoutP, typename SLayoutS,
          typename SLayoutVTma, typename SLayoutY, typename SLayoutSplitY, int kBlockSize,
          int kStage, int kSplitK, int kSplitMinLen>
__global__ void attention_decode_fp8_multistage_ws_smallm_splitk_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaY tma_y,
    const __grid_constant__ TmaSplitY tma_splity,
    TensorQ Q, TensorY Y, TensorSplitY splitY,
    Tout *y_ptr, float *split_y_ptr, float *lse_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr, bool new_kv_included,
    int num_batch, int num_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int lse_pad_heads_per_group, int num_kvcache_blocks,
    int num_seq_max_blocks, int qscale_pad_stride, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ihead_kv = blockIdx.x;
  int ibatch = blockIdx.y;
  int ichunk = blockIdx.z;

  constexpr int kMathThreads = size(TiledMmaQK{}) * kWarpGroupN;
  constexpr int kMathWarps = kMathThreads / 32;
  constexpr int kWarpsPerWrapGroup = 4;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  int num_seq_kvcache, num_seq_kv, num_blocks, num_blocks_per_chunk, num_chunks;
  int num_tile_kv, num_tile_full, num_tile_causal;
  bool is_split, is_last_chunk;

  if (!get_task<kTileN, kBlockSize, kSplitK, kSplitMinLen>(
          num_seq_kvcache_ptr, new_kv_included, num_seq_q, ibatch, ichunk, num_seq_kvcache,
          num_seq_kv, num_chunks, is_split, is_last_chunk, num_blocks, num_blocks_per_chunk,
          num_tile_kv, num_tile_full, num_tile_causal)) {
    return;
  }

  float *lse_batch =
      lse_ptr + ibatch * kSplitK * kWarpGroupN * num_head_k * lse_pad_heads_per_group * num_seq_q +
      ichunk * kWarpGroupN * num_head_k * lse_pad_heads_per_group * num_seq_q +
      ihead_kv * lse_pad_heads_per_group * num_seq_q;

  const int *block_ids =
      block_ids_ptr + ibatch * num_seq_max_blocks + ichunk * num_blocks_per_chunk;

  float kscale = kscale_ptr[0];
  float vscale = vscale_ptr[0];

  auto *qscales = qscale_ptr + ibatch * num_head_q * num_seq_q + ihead_kv * heads_per_group;

  __shared__ uint64_t q_readable;
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_writable[kStage];
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t v_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_vtma = shm_k + cosize(SLayoutK{});
  auto *shm_p = shm_vtma + cosize(SLayoutVTma{});
  auto *shm_max = reinterpret_cast<float *>(shm_p + cosize(SLayoutP{}));
  auto *shm_kvblk_ids =
      reinterpret_cast<int *>(shm_max + kTileM * kWarpsPerWrapGroup * kWarpGroupN);
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);        // Reuse All
  auto *shm_splity = reinterpret_cast<float *>(shm_data);  // Reuse All

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, num_batch));
  auto gSplitY = tma_splity.get_tma_tensor(make_shape(num_dim_v, heads_per_group, num_head_k,
                                                      num_seq_q, kSplitK * kWarpGroupN, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileV>{}, Int<kTileM>{}), make_stride(Int<1>{}, Int<kTileV>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sVTma = make_tensor(make_smem_ptr(shm_vtma), SLayoutVTma{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, seqlenq, head_kv, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head_kv, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head_kv, batch)

  auto tQs = btma_q.partition_D(sQ);     // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);     // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sVTma);  // (TMA, _1, _1)

  // init bar
  if (is_leader_in_block) {
    initialize_barrier(q_readable, 1);
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      initialize_barrier(k_writable[istage], 1);
      initialize_barrier(v_writable[istage], 1);
      initialize_barrier(k_readable[istage], 1);
      initialize_barrier(v_readable[istage], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  // load warpgroup
  if (idx >= kMathThreads) {
    // cutlass::arch::warpgroup_reg_dealloc<24>();
    bool is_leader_in_load = ((iwarp == kMathThreads / 32) && elected);

    if ((heads_per_group == kHeadsPerGroup) || (num_head_q == 4 && num_head_k == 1)) {
      if (is_leader_in_load) {
        // Load Q via TMA
        for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
          cute::copy(tma_q.with(q_readable), tQg(_, 0, _, ihead_kv, iseqq, ibatch),
                     tQs(_, iseqq, _));
        }
        set_barrier_transaction_bytes(q_readable, sizeof(Tin) * cosize(SLayoutQ{}));
      }
    }
  }

  // Load BlockIds
  for (int i = idx; i < num_blocks; i += blockDim.x) {
    shm_kvblk_ids[i] = block_ids[i];
  }
  __syncthreads();

  if (idx >= kMathThreads) {
    idx -= kMathThreads;
    iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);

    constexpr int kBlockPerTileN = kTileN / kBlockSize;

    bool is_leader_in_load = ((iwarp == 0) && elected);
    int phase = 1;
    int iload_tile = 0;

    if (is_leader_in_load) {
      int istage_write = 0;
      // Load Causal KV
#pragma unroll 1
      for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        // load k/scale/v
        load_paged_kv<true, kBlockPerTileN, kBlockSize, kStage, Tin>(
            tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
            ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, itile_seq_kv, istage_write,
            phase);
        advance_stage<kStage>(istage_write, phase);
      }

      // Load Full KV
#pragma unroll 1
      for (int itile_seq_kv = -kStage + 1; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
        if (iload_tile < num_tile_full) {
          load_paged_kv<false, kBlockPerTileN, kBlockSize, kStage, Tin>(
              tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
              ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, iload_tile++,
              istage_write, phase);
          advance_stage<kStage>(istage_write, phase);
        }
      }
    }
  } else {
    // cutlass::arch::warpgroup_reg_alloc<232>();
    // math warpgroup
    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int ilane_in_warpgroup = idx_in_warpgroup % 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);
    bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

    TiledMmaQK tiled_mma_qk;
    TiledMmaSV tiled_mma_sv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx_in_warpgroup);
    auto thr_mma_sv = tiled_mma_sv.get_slice(idx_in_warpgroup);

    auto tKs4r = thr_mma_qk.partition_A(sK);
    auto tQs4r = thr_mma_qk.partition_B(sQ);
    auto tVs4r = thr_mma_sv.partition_A(sVTma(_, _, 0));
    auto tSs4r = thr_mma_sv.partition_B(sS);

    auto tKr = thr_mma_qk.make_fragment_A(tKs4r);  // (MMA, MMA_N, MMA_K)
    auto tQr = thr_mma_qk.make_fragment_B(tQs4r);  // (MMA, MMA_M, MMA_K)
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);  // (MMA, MMA_V, MMA_N)
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);  // (MMA, MMA_V, MMA_N)

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tAttAfp8 = make_tensor_like<Tin>(tAttr);
    auto tYr = thr_mma_sv.partition_fragment_C(gYY);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);
    auto tI_nm = retile_fragment(tI);
    auto tYr_nm = retile_fragment(tYr);

    auto tAttr_nm = retile_fragment(tAttr);
    constexpr int kN = size<0>(tAttr_nm);
    constexpr int kM = size<1>(tAttr_nm);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});
    Tensor qkscales = make_tensor<float>(Int<kM>{});
    Tensor gSoftmaxScale = make_tensor<float>(Int<kM>{});

#pragma unroll
    for (int i = 0; i < kM; i++) {
      int im = get<1>(tI_nm(0, i));
      int iseqq = im / kHeadsPerGroup;
      int iqhead = im % kHeadsPerGroup;
      if (iqhead < heads_per_group) {
        qkscales(i) = qscales[iseqq * num_head_q + iqhead];
      } else {
        qkscales(i) = 1;
      }
    }

    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());
    fill(gSoftmaxScale, one_over_dk_log2e);

    auto tiled_copy_P_r2s = make_tiled_copy_P_interleave<kTileM, Tin>();
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx_in_warpgroup);
    auto tPr4s = thr_copy_P_r2s.retile_S(tAttAfp8);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP(_, _, iwarpgroup));

    auto tiled_copy_VT_s2r = make_tiled_copy_V_interleave_trans<Tin>();
    auto thr_copy_VT_s2r = tiled_copy_VT_s2r.get_slice(idx_in_warpgroup);
    auto tVTs4r = thr_copy_VT_s2r.partition_S(sVTma);
    auto tVTr4s = make_fragment_like(thr_copy_VT_s2r.partition_D(sVTma(_, _, 0)));

    auto v_for_trans = recast<uint32_t>(tVTr4s);
    auto vt_for_trans = recast<uint32_t>(
        make_tensor(tVr.data(), left_inverse(tVr.layout()).compose(tVTr4s.layout())));

    using R2SCopyAtomSplitY = Copy_Atom<UniversalCopy<uint32_t>, float>;
    auto tiled_copy_SplitY_r2s = make_tiled_copy_Y_interleave<kTileM>(R2SCopyAtomSplitY{});

    using R2SCopyAtomY = Copy_Atom<UniversalCopy<uint16_t>, Tout>;
    auto tiled_copy_Y_r2s = make_tiled_copy_Y_interleave<kTileM>(R2SCopyAtomY{});

    clear(tYr);

    tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

    if ((heads_per_group == kHeadsPerGroup) || (num_head_q == 4 && num_head_k == 1)) {
      wait_barrier(q_readable, 0);
    } else {
      // if not using TMA, each math warpgroup independently loads Q from GMEM to smem.
      // sQ is shared across warpgroups (single allocation), so both warpgroups write the same
      // data — this is safe because the data is identical (Q is read-only).
      // Use idx_in_warpgroup and per-warpgroup thread count to avoid cross-warpgroup strides.
      constexpr int kThreadsPerWG = kMathThreads / kWarpGroupN;
      load_q_group_direct_to_smem_fp8<Tin, kTileK>(Q, sQ, ihead_kv, ibatch, heads_per_group,
                                                    num_seq_q, kHeadsPerGroup, idx_in_warpgroup,
                                                    kThreadsPerWG);
      syncwarpgroup(iwarpgroup);
    }

    int phase = 0;
    int istage_read = iwarpgroup;
    shm_max += iwarpgroup * kTileM * kWarpsPerWrapGroup;
    lse_batch += iwarpgroup * num_head_k * lse_pad_heads_per_group * num_seq_q;
    bool warpgroup_computed = false;

#pragma unroll
    for (int i = 0; i < kM; i++) {
      qkscales(i) *= kscale;
    }
    // compute casual
#pragma unroll 1
    for (int itile_seq_kv = num_tile_full + iwarpgroup; itile_seq_kv < num_tile_kv;
         itile_seq_kv += kWarpGroupN) {
      warpgroup_computed = true;
      wait_barrier(k_readable[istage_read], phase);

      // P = QK
      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      // do causal mask (also masks out-of-range Q heads via apply_casual_mask_with_scale)
      apply_casual_mask_with_scale<kTileN, kHeadsPerGroup>(tAttr_nm, tI_nm, qkscales, itile_seq_kv,
                                                           num_seq_kvcache, num_seq_kv);
      // mask invalid Q-head slots when actual heads_per_group < kHeadsPerGroup.
      if (heads_per_group < kHeadsPerGroup) {
#pragma unroll
        for (int im = 0; im < kM; ++im) {
          if (cute::get<1>(tI_nm(0, im)) % kHeadsPerGroup >= heads_per_group) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
            }
          }
        }
      }

      // online softmax
      online_softmax<true, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max, iwarpgroup,
                                   iwarp_in_warpgroup, ilane_in_warpgroup);

      // tAttfp32 => fp8
      cast_fp32reg<Tin>(tAttr_nm, tAttAfp8);

      // permute P
      permute_p(tAttAfp8, 0x7520);

      // P reg to smem
      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);
      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      // V smem to reg
      cute::copy(tiled_copy_VT_s2r, tVTs4r(_, _, _, istage_read), tVTr4s);

      // permute v and sv mma
      permute_v_sv_gemm(tiled_mma_sv, tSr, tVr, tYr, v_for_trans, vt_for_trans, iwarpgroup);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage, kWarpGroupN>(istage_read, phase);

      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr);
    }

#pragma unroll
    for (int i = 0; i < kM; i++) {
      gSoftmaxScale(i) *= qkscales(i);
    }

    // compute full
#pragma unroll 1
    for (int itile_seq_kv = (kWarpGroupN - num_tile_causal + iwarpgroup) % kWarpGroupN;
         itile_seq_kv < num_tile_full; itile_seq_kv += kWarpGroupN) {
      warpgroup_computed = true;
      wait_barrier(k_readable[istage_read], phase);

      // P = QK
      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      // mask invalid Q-head slots when actual heads_per_group < kHeadsPerGroup.
      if (heads_per_group < kHeadsPerGroup) {
#pragma unroll
        for (int im = 0; im < kM; ++im) {
          if (cute::get<1>(tI_nm(0, im)) % kHeadsPerGroup >= heads_per_group) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
            }
          }
        }
      }

      // online softmax
      online_softmax<false, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max,
                                    iwarpgroup, iwarp_in_warpgroup, ilane_in_warpgroup);

      // Y = PV
      cast_fp32reg<Tin>(tAttr_nm, tAttAfp8);

      // permute P
      permute_p(tAttAfp8, 0x7520);

      // P reg to smem
      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);
      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      // V smem to reg
      cute::copy(tiled_copy_VT_s2r, tVTs4r(_, _, _, istage_read), tVTr4s);

      // permute v and sv mma
      permute_v_sv_gemm(tiled_mma_sv, tSr, tVr, tYr, v_for_trans, vt_for_trans, iwarpgroup);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage, kWarpGroupN>(istage_read, phase);

      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr);
    }

    if (warpgroup_computed) {
      // final online softmax
      final_online_softmax<kTileM>(tYr_nm, gSum, shm_max, iwarpgroup, iwarp_in_warpgroup,
                                   ilane_in_warpgroup);
    }

#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      tYr(i) *= vscale;
    }

    bar_sync<kWarpGroupN * 128>(kWarpGroupN);

    const bool use_tma_store =
        (heads_per_group == kHeadsPerGroup) || (num_head_q == 4 && num_head_k == 1);

    if constexpr (kWarpGroupN == 1) {
      if (!is_split) {
        auto tYr_bf16 = make_tensor_like<Tout>(tYr);
        // to bfloat16
        cast_fp32reg<Tout>(tYr, tYr_bf16);
        if (use_tma_store) {
          store_output<false, 1>(tiled_copy_Y_r2s, tma_y, tYr_bf16, sY, gY, ihead_kv, ibatch, 0,
                                 num_seq_q, idx, iwarpgroup, is_leader_in_warpgroup);
        } else {
          // Write bf16 result back through sY then direct GMEM store
          auto thr_copy_y = tiled_copy_Y_r2s.get_slice(idx_in_warpgroup);
          auto tYr4s = thr_copy_y.retile_S(tYr_bf16);
          auto tYs4r = thr_copy_y.partition_D(sY(_, _, iwarpgroup));
          cute::copy(tiled_copy_Y_r2s, tYr4s, tYs4r);
          bar_sync<128>(1);
          // kWarpGroupN==1 here, so iwarpgroup==0 and idx_in_warpgroup==idx
          store_sY_to_gmem_bf16_fp8<Tout, kTileV>(sY, Y, ihead_kv, ibatch, heads_per_group,
                                                   num_seq_q, kHeadsPerGroup, iwarpgroup,
                                                   idx_in_warpgroup, kMathThreads / kWarpGroupN);
        }
        return;
      }
    }
    if (use_tma_store) {
      store_output<true, kWarpGroupN>(tiled_copy_SplitY_r2s, tma_splity, tYr, sSplitY, gSplitY,
                                      ihead_kv, ibatch, ichunk, num_seq_q, idx_in_warpgroup,
                                      iwarpgroup, is_leader_in_warpgroup);
    } else {
      // Write float result back through sSplitY then direct GMEM store.
      // Each warpgroup writes to its own slice sSplitY(_, _, iwarpgroup), then stores to
      // splitY at slot (ichunk * kWarpGroupN + iwarpgroup) to match the TMA path convention.
      auto thr_copy_sy = tiled_copy_SplitY_r2s.get_slice(idx_in_warpgroup);
      auto tYr4s = thr_copy_sy.retile_S(tYr);
      auto tYs4r = thr_copy_sy.partition_D(sSplitY(_, _, iwarpgroup));
      cute::copy(tiled_copy_SplitY_r2s, tYr4s, tYs4r);
      bar_sync<128>(1);
      const int ichunk_wg = ichunk * kWarpGroupN + iwarpgroup;
      store_sSplitY_to_gmem_float_fp8(sSplitY, splitY, ihead_kv, ichunk_wg, ibatch,
                                      heads_per_group, num_seq_q, kHeadsPerGroup, num_dim_v,
                                      iwarpgroup, idx_in_warpgroup, kMathThreads / kWarpGroupN);
    }
    store_lse(lse_batch, gMax, gSum, heads_per_group, ilane_in_warpgroup, iwarp_in_warpgroup);

    auto *split_flag = split_flag_ptr + ibatch * num_head_k + ihead_kv;

    tma_store_wait<0>();
    __threadfence();
    bar_sync<kWarpGroupN * 128>(kWarpGroupN);

    if (idx_in_warpgroup == 0) {
      atomicAdd(split_flag, 1);
    }

    if (is_last_chunk) {
      while (load_global_volatile(split_flag) != (ichunk + 1) * kWarpGroupN) {
      }

      splitk_reduce<__nv_bfloat16, kTileV, kSplitK * kWarpGroupN, kMathWarps>(
          y_ptr, lse_ptr, split_y_ptr, num_chunks * kWarpGroupN, num_seq_q, num_head_q, num_head_k,
          heads_per_group, lse_pad_heads_per_group, ihead_kv, ibatch, iwarp, ilane_in_warpgroup);
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SMALLM_SPLITK_FP8_KERNELS_CUH_
