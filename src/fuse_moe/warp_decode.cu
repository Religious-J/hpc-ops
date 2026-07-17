// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <type_traits>

#include "src/fuse_moe/fuse_moe.h"
#include "src/fuse_moe/warp_decode.h"
#include "src/group_gemm/cp_async/group_gemm.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kElementsPerPack = 4;

__device__ __forceinline__ float4 load_fp8x4(const __nv_fp8_e4m3 *ptr) {
  return static_cast<float4>(*reinterpret_cast<const __nv_fp8x4_e4m3 *>(ptr));
}

template <bool kUseBFloat16PrecisionMultiply>
__global__ void split_partial_act_quant_kernel(
    __nv_fp8_e4m3 *__restrict__ output_ptr,
    const __nv_bfloat16 *__restrict__ partial_ptr,
    const float *__restrict__ scale_ptr, int num_routes, int num_splits,
    int intermediate_size) {
  cudaGridDependencySynchronize();

  constexpr int kVectorSize = 8;
  const int num_vectors = num_routes * intermediate_size / kVectorSize;
  for (int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
       vector_idx < num_vectors; vector_idx += blockDim.x * gridDim.x) {
    const int element_idx = vector_idx * kVectorSize;
    const int route = element_idx / intermediate_size;
    const int col = element_idx - route * intermediate_size;
    float gate[kVectorSize] = {};
    float up[kVectorSize] = {};

#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      const uint64_t base =
          (static_cast<uint64_t>(route) * num_splits + split) *
          (2 * intermediate_size);
#pragma unroll
      for (int i = 0; i < kVectorSize; ++i) {
        gate[i] += __bfloat162float(partial_ptr[base + col + i]);
        up[i] += __bfloat162float(
            partial_ptr[base + intermediate_size + col + i]);
      }
    }

    float activated[kVectorSize];
#pragma unroll
    for (int i = 0; i < kVectorSize; ++i) {
      gate[i] = __bfloat162float(__float2bfloat16_rn(gate[i]));
      up[i] = __bfloat162float(__float2bfloat16_rn(up[i]));
      if constexpr (kUseBFloat16PrecisionMultiply) {
        auto silu_bf16 = __float2bfloat16_rn(silu(gate[i]));
        auto up_bf16 = __float2bfloat16_rn(up[i]);
        activated[i] = __bfloat162float(silu_bf16 * up_bf16) * scale_ptr[0];
      } else {
        activated[i] = silu(gate[i]) * up[i] * scale_ptr[0];
      }
    }
    auto *output = output_ptr + element_idx;
    *reinterpret_cast<__nv_fp8x4_e4m3 *>(output) =
        __nv_fp8x4_e4m3(*reinterpret_cast<float4 *>(&activated[0]));
    *reinterpret_cast<__nv_fp8x4_e4m3 *>(output + 4) =
        __nv_fp8x4_e4m3(*reinterpret_cast<float4 *>(&activated[4]));
  }

  // Some threads can have no vector to write. Let natural CTA completion
  // satisfy PDL instead of allowing an idle thread to trigger it early.
}

}  // namespace kernels

void split_partial_act_quant_async(
    void *output_ptr, const void *partial_ptr, const void *scale_ptr,
    int num_routes, int num_splits, int intermediate_size,
    bool use_bf16_mul, cudaStream_t stream) {
  constexpr int kThreads = 256;
  constexpr int kVectorSize = 8;
  const int num_vectors = num_routes * intermediate_size / kVectorSize;
  const int grid = (num_vectors + kThreads - 1) / kThreads;

  auto launch = [&](auto bf16_mul_tag) {
    constexpr bool kUseBFloat16PrecisionMultiply = decltype(bf16_mul_tag)::value;
    auto kernel =
        kernels::split_partial_act_quant_kernel<kUseBFloat16PrecisionMultiply>;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(
        &cfg, kernel, static_cast<__nv_fp8_e4m3 *>(output_ptr),
        static_cast<const __nv_bfloat16 *>(partial_ptr),
        static_cast<const float *>(scale_ptr), num_routes, num_splits,
        intermediate_size);
  };
  if (use_bf16_mul) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

void fuse_moe_warp_decode_mma_async(
    void *output_ptr, const void *input_ptr, void *intermediate_ptr,
    const void *gate_up_weight_ptr, const void *gate_up_scale_ptr,
    const void *act_and_mul_scale_ptr, const void *down_weight_ptr,
    const void *down_scale_ptr, const void *topk_ids_ptr, const void *topk_scale_ptr,
    const void *shared_output_ptr, int num_seq, int hidden_size, int intermediate_size,
    int num_topk, int num_splits, int num_expert_local, int rank_ep, bool use_bf16_mul,
    cudaStream_t stream) {
  const int num_routes = num_seq * num_topk;
  auto *workspace = static_cast<uint8_t *>(intermediate_ptr);
  auto *gate_up_partials = reinterpret_cast<__nv_bfloat16 *>(workspace);
  workspace += static_cast<size_t>(num_routes) * num_splits * 2 * intermediate_size *
               sizeof(__nv_bfloat16);
  auto *down_input = reinterpret_cast<__nv_fp8_e4m3 *>(workspace);
  workspace += static_cast<size_t>(num_routes) * intermediate_size *
               sizeof(__nv_fp8_e4m3);
  auto *down_output = reinterpret_cast<__nv_bfloat16 *>(workspace);

  group_gemm_cp_async::group_gemm_fp8_route_splitk_async(
      gate_up_partials, input_ptr, gate_up_weight_ptr, gate_up_scale_ptr,
      topk_ids_ptr, num_routes, num_topk, 2 * intermediate_size, hidden_size,
      num_splits, num_expert_local, rank_ep, stream);

  split_partial_act_quant_async(
      down_input, gate_up_partials, act_and_mul_scale_ptr, num_routes,
      num_splits, intermediate_size, use_bf16_mul, stream);

  group_gemm_cp_async::group_gemm_fp8_route_async(
      down_output, down_input, down_weight_ptr, down_scale_ptr, topk_ids_ptr,
      num_routes, num_topk, hidden_size, intermediate_size, num_expert_local,
      rank_ep, /*input_is_token=*/false, stream);

  reduce_async(output_ptr, down_output, /*topk_pos_ptr=*/nullptr, topk_scale_ptr,
               shared_output_ptr, num_routes, num_seq, hidden_size, num_topk,
               /*use_pdl=*/true, stream);
}

namespace kernels {

constexpr int kBlockwiseQuantSize = 128;

template <int kNumTopk>
__global__ void warp_decode_blockwise_gate_up_kernel(
    const __nv_fp8_e4m3 *__restrict__ input_ptr,
    const float *__restrict__ input_scale_ptr,
    const __nv_fp8_e4m3 *__restrict__ gate_up_weight_ptr,
    const float *__restrict__ gate_up_weight_scale_ptr,
    const int *__restrict__ topk_ids_ptr, float *__restrict__ activated_ptr,
    int hidden_size, int intermediate_size,
    int num_expert_local, int scale_lastdim_pad4, int start_expert) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int neuron = blockIdx.x * kWarpsPerBlock + warp;
  const int route = blockIdx.y;

  const int expert = topk_ids_ptr[route];
  const int local_expert = expert - start_expert;
  if (local_expert < 0 || local_expert >= num_expert_local) {
    return;
  }

  const int token = route / kNumTopk;
  const auto *global_input_row =
      input_ptr + static_cast<int64_t>(token) * hidden_size;
  const auto *input_row = global_input_row;
  extern __shared__ uint32_t shared_input_packs[];
  const int num_input_packs = hidden_size / kElementsPerPack;
  const auto *global_input_packs =
      reinterpret_cast<const uint32_t *>(global_input_row);
  for (int pack = threadIdx.x; pack < num_input_packs; pack += blockDim.x) {
    shared_input_packs[pack] = global_input_packs[pack];
  }
  __syncthreads();
  input_row = reinterpret_cast<const __nv_fp8_e4m3 *>(shared_input_packs);

  if (neuron >= intermediate_size) {
    return;
  }

  const int num_k_blocks = hidden_size / kBlockwiseQuantSize;
  const int64_t expert_weight_stride =
      static_cast<int64_t>(2) * intermediate_size * hidden_size;
  const auto *expert_weight =
      gate_up_weight_ptr +
      static_cast<int64_t>(local_expert) * expert_weight_stride;
  const int64_t expert_scale_stride =
      static_cast<int64_t>((2 * intermediate_size + kBlockwiseQuantSize - 1) /
                           kBlockwiseQuantSize) *
      scale_lastdim_pad4;
  const auto *expert_scale = gate_up_weight_scale_ptr +
                             static_cast<int64_t>(local_expert) * expert_scale_stride;
  const auto *token_scale = input_scale_ptr +
                            static_cast<int64_t>(token) * num_k_blocks;

  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
    const int block_start = k_block * kBlockwiseQuantSize;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    const auto *gate_row =
        expert_weight + static_cast<int64_t>(neuron) * hidden_size;
    const auto *up_row =
        gate_row + static_cast<int64_t>(intermediate_size) * hidden_size;
    for (int offset = lane * kElementsPerPack;
         offset < kBlockwiseQuantSize;
         offset += kWarpSize * kElementsPerPack) {
      const int col = block_start + offset;
      const float4 x = load_fp8x4(input_row + col);
      const float4 gate = load_fp8x4(gate_row + col);
      const float4 up = load_fp8x4(up_row + col);
      gate_partial = fmaf(x.x, gate.x, gate_partial);
      gate_partial = fmaf(x.y, gate.y, gate_partial);
      gate_partial = fmaf(x.z, gate.z, gate_partial);
      gate_partial = fmaf(x.w, gate.w, gate_partial);
      up_partial = fmaf(x.x, up.x, up_partial);
      up_partial = fmaf(x.y, up.y, up_partial);
      up_partial = fmaf(x.z, up.z, up_partial);
      up_partial = fmaf(x.w, up.w, up_partial);
    }
    const float activation_scale = token_scale[k_block];
    gate_partial = warp_reduce_sum_xor(gate_partial);
    up_partial = warp_reduce_sum_xor(up_partial);
    const int gate_scale_row = neuron / kBlockwiseQuantSize;
    const int up_scale_row =
        (intermediate_size + neuron) / kBlockwiseQuantSize;
    gate_sum =
        fmaf(gate_partial,
             activation_scale *
                 expert_scale[gate_scale_row * scale_lastdim_pad4 + k_block],
             gate_sum);
    up_sum =
        fmaf(up_partial,
             activation_scale *
                 expert_scale[up_scale_row * scale_lastdim_pad4 + k_block],
             up_sum);
  }

  if (lane == 0) {
    const float gate = __bfloat162float(__float2bfloat16_rn(gate_sum));
    const float up = __bfloat162float(__float2bfloat16_rn(up_sum));
    activated_ptr[static_cast<int64_t>(route) * intermediate_size + neuron] =
        silu(gate) * up;
  }
}

template <int kNumTopk>
__global__ void warp_decode_blockwise_quant_kernel(
    const float *__restrict__ activated_ptr,
    __nv_fp8_e4m3 *__restrict__ quantized_ptr,
    float *__restrict__ quantized_scale_ptr,
    const int *__restrict__ topk_ids_ptr, int intermediate_size,
    int num_expert_local, int start_expert) {
  const int route = blockIdx.y;
  const int block = blockIdx.x;
  const int lane = threadIdx.x;
  const int col = block * kBlockwiseQuantSize + lane;
  const int64_t index = static_cast<int64_t>(route) * intermediate_size + col;
  const int num_blocks =
      (intermediate_size + kBlockwiseQuantSize - 1) / kBlockwiseQuantSize;
  const int local_expert = topk_ids_ptr[route] - start_expert;
  if (local_expert < 0 || local_expert >= num_expert_local) {
    if (col < intermediate_size) {
      quantized_ptr[index] = __nv_fp8_e4m3(0.0f);
    }
    if (lane == 0) {
      quantized_scale_ptr[static_cast<int64_t>(route) * num_blocks + block] =
          0.0f;
    }
    return;
  }

  const float value =
      col < intermediate_size ? activated_ptr[index] : 0.0f;
  float max_value = warp_reduce_max_xor(fabsf(value));
  __shared__ float warp_max[4];
  if ((lane & 31) == 0) {
    warp_max[lane / 32] = max_value;
  }
  __syncthreads();
  if (lane < 32) {
    max_value = lane < 4 ? warp_max[lane] : 0.0f;
    max_value = warp_reduce_max_xor(max_value);
  }
  __shared__ float inverse_scale;
  if (lane == 0) {
    const float scale = max_value / 448.0f;
    quantized_scale_ptr[static_cast<int64_t>(route) * num_blocks + block] = scale;
    inverse_scale = 1.0f / (scale + 1e-8f);
  }
  __syncthreads();
  if (col < intermediate_size) {
    quantized_ptr[index] = __nv_fp8_e4m3(value * inverse_scale);
  }
}

template <int kNumTopk, bool kHasTail>
__global__ void warp_decode_blockwise_down_kernel(
    __nv_bfloat16 *__restrict__ output_ptr,
    const __nv_fp8_e4m3 *__restrict__ quantized_ptr,
    const float *__restrict__ quantized_scale_ptr,
    const __nv_fp8_e4m3 *__restrict__ down_weight_ptr,
    const float *__restrict__ down_weight_scale_ptr,
    const int *__restrict__ topk_ids_ptr,
    const float *__restrict__ topk_scale_ptr,
    const __nv_bfloat16 *__restrict__ shared_output_ptr, int hidden_size,
    int intermediate_size, int num_expert_local, int scale_lastdim_pad4,
    int start_expert) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int output_col = blockIdx.x * warps_per_block + warp;
  const int token = blockIdx.y;

  extern __shared__ uint32_t shared_storage[];
  const int activation_packs =
      kNumTopk * intermediate_size / kElementsPerPack;
  const auto *global_activation_packs = reinterpret_cast<const uint32_t *>(
      quantized_ptr + static_cast<int64_t>(token) * kNumTopk * intermediate_size);
  for (int pack = threadIdx.x; pack < activation_packs; pack += blockDim.x) {
    shared_storage[pack] = global_activation_packs[pack];
  }
  float *shared_scales = reinterpret_cast<float *>(shared_storage + activation_packs);
  const int num_k_blocks =
      kHasTail ? (intermediate_size + kBlockwiseQuantSize - 1) /
                     kBlockwiseQuantSize
               : intermediate_size / kBlockwiseQuantSize;
  const int num_scales = kNumTopk * num_k_blocks;
  const auto *global_scales = quantized_scale_ptr +
                              static_cast<int64_t>(token) * kNumTopk * num_k_blocks;
  for (int index = threadIdx.x; index < num_scales; index += blockDim.x) {
    shared_scales[index] = global_scales[index];
  }
  __syncthreads();

  if (output_col >= hidden_size) {
    return;
  }

  const int num_n_blocks = hidden_size / kBlockwiseQuantSize;
  const int output_scale_row = output_col / kBlockwiseQuantSize;
  const int64_t expert_scale_stride =
      static_cast<int64_t>(num_n_blocks) * scale_lastdim_pad4;
  float output_sum = 0.0f;

#pragma unroll
  for (int topk = 0; topk < kNumTopk; ++topk) {
    const int route = token * kNumTopk + topk;
    const int local_expert = topk_ids_ptr[route] - start_expert;
    if (local_expert < 0 || local_expert >= num_expert_local) {
      continue;
    }
    const auto *activation_row =
        reinterpret_cast<const __nv_fp8_e4m3 *>(shared_storage) +
        topk * intermediate_size;
    const auto *weight_row =
        down_weight_ptr +
        (static_cast<int64_t>(local_expert) * hidden_size + output_col) *
            intermediate_size;
    const auto *weight_scale = down_weight_scale_ptr +
                               static_cast<int64_t>(local_expert) * expert_scale_stride +
                               output_scale_row * scale_lastdim_pad4;
    float expert_sum = 0.0f;
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
      const int col = k_block * kBlockwiseQuantSize +
                      lane * kElementsPerPack;
      float partial = 0.0f;
      if constexpr (kHasTail) {
        if (col < intermediate_size) {
          const float4 activation = load_fp8x4(activation_row + col);
          const float4 weight = load_fp8x4(weight_row + col);
          partial = fmaf(activation.x, weight.x, partial);
          partial = fmaf(activation.y, weight.y, partial);
          partial = fmaf(activation.z, weight.z, partial);
          partial = fmaf(activation.w, weight.w, partial);
        }
      } else {
        const float4 activation = load_fp8x4(activation_row + col);
        const float4 weight = load_fp8x4(weight_row + col);
        partial = fmaf(activation.x, weight.x, partial);
        partial = fmaf(activation.y, weight.y, partial);
        partial = fmaf(activation.z, weight.z, partial);
        partial = fmaf(activation.w, weight.w, partial);
      }
      partial = warp_reduce_sum_xor(partial);
      expert_sum = fmaf(partial,
                        shared_scales[topk * num_k_blocks + k_block] *
                            weight_scale[k_block],
                        expert_sum);
    }
    const float rounded_expert_sum =
        __bfloat162float(__float2bfloat16_rn(expert_sum));
    output_sum = fmaf(rounded_expert_sum, topk_scale_ptr[route], output_sum);
  }

  if (lane == 0) {
    const int64_t output_index =
        static_cast<int64_t>(token) * hidden_size + output_col;
    if (shared_output_ptr != nullptr) {
      output_sum += __bfloat162float(shared_output_ptr[output_index]);
    }
    output_ptr[output_index] = __float2bfloat16_rn(output_sum);
  }
}

}  // namespace kernels

void fuse_moe_warp_decode_blockwise_async(
    void *output_ptr, const void *input_ptr, const void *input_scale_ptr,
    void *activated_ptr, void *quantized_ptr, void *quantized_scale_ptr,
    const void *gate_up_weight_ptr, const void *gate_up_weight_scale_ptr,
    const void *down_weight_ptr, const void *down_weight_scale_ptr,
    const void *topk_ids_ptr, const void *topk_scale_ptr,
    const void *shared_output_ptr, int num_tokens, int hidden_size,
    int intermediate_size, int num_topk, int num_expert_local,
    int gate_up_weight_scale_lastdim_pad4,
    int down_weight_scale_lastdim_pad4, int rank_ep, cudaStream_t stream) {
  constexpr int kThreads =
      kernels::kWarpsPerBlock * kernels::kWarpSize;
  const int start_expert = rank_ep * num_expert_local;
  const dim3 gate_grid(
      (intermediate_size + kernels::kWarpsPerBlock - 1) /
          kernels::kWarpsPerBlock,
      num_tokens * num_topk);
  const size_t gate_shared_bytes =
      static_cast<size_t>(hidden_size) * sizeof(__nv_fp8_e4m3);
  const int num_intermediate_blocks =
      (intermediate_size + kernels::kBlockwiseQuantSize - 1) /
      kernels::kBlockwiseQuantSize;
  const dim3 quant_grid(num_intermediate_blocks, num_tokens * num_topk);
  const size_t down_shared_bytes =
      static_cast<size_t>(num_topk) * intermediate_size * sizeof(__nv_fp8_e4m3) +
      static_cast<size_t>(num_topk) * num_intermediate_blocks * sizeof(float);

  if (num_topk == 8) {
    kernels::warp_decode_blockwise_gate_up_kernel<8>
        <<<gate_grid, kThreads, gate_shared_bytes, stream>>>(
            static_cast<const __nv_fp8_e4m3 *>(input_ptr),
            static_cast<const float *>(input_scale_ptr),
            static_cast<const __nv_fp8_e4m3 *>(gate_up_weight_ptr),
            static_cast<const float *>(gate_up_weight_scale_ptr),
            static_cast<const int *>(topk_ids_ptr),
            static_cast<float *>(activated_ptr), hidden_size,
            intermediate_size, num_expert_local,
            gate_up_weight_scale_lastdim_pad4, start_expert);
    kernels::warp_decode_blockwise_quant_kernel<8>
        <<<quant_grid, kernels::kBlockwiseQuantSize, 0, stream>>>(
            static_cast<const float *>(activated_ptr),
            static_cast<__nv_fp8_e4m3 *>(quantized_ptr),
            static_cast<float *>(quantized_scale_ptr),
            static_cast<const int *>(topk_ids_ptr), intermediate_size,
            num_expert_local, start_expert);
    const int down_warps = num_tokens == 1 ? 16 : 8;
    const dim3 down_grid((hidden_size + down_warps - 1) / down_warps,
                         num_tokens);
    if (intermediate_size % kernels::kBlockwiseQuantSize != 0) {
      kernels::warp_decode_blockwise_down_kernel<8, true>
          <<<down_grid, down_warps * kernels::kWarpSize,
             down_shared_bytes, stream>>>(
              static_cast<__nv_bfloat16 *>(output_ptr),
              static_cast<const __nv_fp8_e4m3 *>(quantized_ptr),
              static_cast<const float *>(quantized_scale_ptr),
              static_cast<const __nv_fp8_e4m3 *>(down_weight_ptr),
              static_cast<const float *>(down_weight_scale_ptr),
              static_cast<const int *>(topk_ids_ptr),
              static_cast<const float *>(topk_scale_ptr),
              static_cast<const __nv_bfloat16 *>(shared_output_ptr), hidden_size,
              intermediate_size, num_expert_local,
              down_weight_scale_lastdim_pad4, start_expert);
    } else {
      kernels::warp_decode_blockwise_down_kernel<8, false>
          <<<down_grid, down_warps * kernels::kWarpSize,
             down_shared_bytes, stream>>>(
              static_cast<__nv_bfloat16 *>(output_ptr),
              static_cast<const __nv_fp8_e4m3 *>(quantized_ptr),
              static_cast<const float *>(quantized_scale_ptr),
              static_cast<const __nv_fp8_e4m3 *>(down_weight_ptr),
              static_cast<const float *>(down_weight_scale_ptr),
              static_cast<const int *>(topk_ids_ptr),
              static_cast<const float *>(topk_scale_ptr),
              static_cast<const __nv_bfloat16 *>(shared_output_ptr), hidden_size,
              intermediate_size, num_expert_local,
              down_weight_scale_lastdim_pad4, start_expert);
    }
  }
}

}  // namespace fuse_moe
}  // namespace hpc
