#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kWarpSize = 32;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float value) {
  static_assert(BLOCK_SIZE % kWarpSize == 0);
  __shared__ float warp_sums[BLOCK_SIZE / kWarpSize];
  __shared__ float block_sum;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    float warp_value = lane < BLOCK_SIZE / kWarpSize ? warp_sums[lane] : 0.0f;
    warp_value = warp_reduce_sum(warp_value);
    if (lane == 0) {
      block_sum = warp_value;
    }
  }
  __syncthreads();
  return block_sum;
}

// V1: intentionally simple shared-memory tree reduction.
template <typename scalar_t, int BLOCK_SIZE>
__global__ void rms_norm_shared_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t hidden_size,
    float epsilon) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t offset = row * hidden_size;

  float sum_sq = 0.0f;
  for (int64_t col = tid; col < hidden_size; col += BLOCK_SIZE) {
    const float value = to_float(input[offset + col]);
    sum_sq += value * value;
  }
  partial[tid] = sum_sq;
  __syncthreads();

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + epsilon);
  for (int64_t col = tid; col < hidden_size; col += BLOCK_SIZE) {
    const float value = to_float(input[offset + col]);
    const float scale = to_float(weight[col]);
    output[offset + col] = from_float<scalar_t>(value * inv_rms * scale);
  }
}

// V2: hierarchical warp-shuffle reduction; only warp leaders use shared memory.
template <typename scalar_t, int BLOCK_SIZE>
__global__ void rms_norm_warp_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t hidden_size,
    float epsilon) {
  const int64_t row = blockIdx.x;
  const int64_t offset = row * hidden_size;

  float sum_sq = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
    const float value = to_float(input[offset + col]);
    sum_sq += value * value;
  }
  const float inv_rms = rsqrtf(
      block_reduce_sum<BLOCK_SIZE>(sum_sq) / static_cast<float>(hidden_size) + epsilon);

  for (int64_t col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
    const float value = to_float(input[offset + col]);
    const float scale = to_float(weight[col]);
    output[offset + col] = from_float<scalar_t>(value * inv_rms * scale);
  }
}

// V2 fused generic path. It writes the residual result in pass one and reads it
// in pass two. V3 below removes that re-read for the Qwen3-0.6B hidden size.
template <typename scalar_t, int BLOCK_SIZE>
__global__ void fused_add_rms_norm_warp_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    int64_t hidden_size,
    float epsilon) {
  const int64_t row = blockIdx.x;
  const int64_t offset = row * hidden_size;

  float sum_sq = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
    const int64_t index = offset + col;
    const scalar_t rounded = from_float<scalar_t>(
        to_float(input[index]) + to_float(residual[index]));
    const float value = to_float(rounded);
    residual_output[index] = rounded;
    sum_sq += value * value;
  }
  const float inv_rms = rsqrtf(
      block_reduce_sum<BLOCK_SIZE>(sum_sq) / static_cast<float>(hidden_size) + epsilon);

  for (int64_t col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
    const int64_t index = offset + col;
    const float value = to_float(residual_output[index]);
    output[index] = from_float<scalar_t>(value * inv_rms * to_float(weight[col]));
  }
}

// V3: SM120/Qwen3-0.6B fast path. Four elements per thread remain in
// registers across the block reduction, avoiding an intermediate re-read.
template <typename scalar_t>
__global__ void fused_add_rms_norm_h1024_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    float epsilon) {
  constexpr int kBlockSize = 256;
  constexpr int kItemsPerThread = 4;
  constexpr int kHiddenSize = kBlockSize * kItemsPerThread;
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * kHiddenSize;
  float values[kItemsPerThread];
  float sum_sq = 0.0f;

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int col = threadIdx.x + item * kBlockSize;
    const int64_t index = offset + col;
    values[item] = to_float(from_float<scalar_t>(
        to_float(input[index]) + to_float(residual[index])));
    sum_sq += values[item] * values[item];
  }

  const float inv_rms = rsqrtf(
      block_reduce_sum<kBlockSize>(sum_sq) / static_cast<float>(kHiddenSize) + epsilon);

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int col = threadIdx.x + item * kBlockSize;
    const int64_t index = offset + col;
    residual_output[index] = from_float<scalar_t>(values[item]);
    output[index] = from_float<scalar_t>(values[item] * inv_rms * to_float(weight[col]));
  }
}

template <typename scalar_t>
struct alignas(16) Pack128 {
  static constexpr int kItems = 16 / sizeof(scalar_t);
  scalar_t values[kItems];
};

// V4: one aligned 128-bit transaction per input/residual/weight/output pack.
// The number of threads adapts to dtype so a block still covers 1024 values.
template <typename scalar_t>
__global__ void fused_add_rms_norm_h1024_packed_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    float epsilon) {
  using Pack = Pack128<scalar_t>;
  constexpr int kItems = Pack::kItems;
  constexpr int kHiddenSize = 1024;
  constexpr int kBlockSize = kHiddenSize / kItems;
  const int64_t pack_offset = static_cast<int64_t>(blockIdx.x) * kBlockSize;

  const Pack input_pack = reinterpret_cast<const Pack*>(input)[pack_offset + threadIdx.x];
  const Pack residual_pack =
      reinterpret_cast<const Pack*>(residual)[pack_offset + threadIdx.x];
  const Pack weight_pack = reinterpret_cast<const Pack*>(weight)[threadIdx.x];
  float values[kItems];
  float sum_sq = 0.0f;

#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    values[item] = to_float(from_float<scalar_t>(
        to_float(input_pack.values[item]) + to_float(residual_pack.values[item])));
    sum_sq += values[item] * values[item];
  }

  const float inv_rms = rsqrtf(
      block_reduce_sum<kBlockSize>(sum_sq) / static_cast<float>(kHiddenSize) + epsilon);
  Pack output_pack;
  Pack residual_output_pack;

#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    residual_output_pack.values[item] = from_float<scalar_t>(values[item]);
    output_pack.values[item] = from_float<scalar_t>(
        values[item] * inv_rms * to_float(weight_pack.values[item]));
  }
  reinterpret_cast<Pack*>(output)[pack_offset + threadIdx.x] = output_pack;
  reinterpret_cast<Pack*>(residual_output)[pack_offset + threadIdx.x] = residual_output_pack;
}

// Qwen3 fast path: fuse per-head Q/K RMSNorm with half-split RoPE. One
// 64-thread block owns one 128-wide head; every thread keeps the two rotary
// components in registers across the reduction and rotation.
template <typename scalar_t>
__global__ void fused_qk_rms_norm_rope_h128_block64_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ query_weight,
    const scalar_t* __restrict__ key_weight,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    int64_t query_heads,
    int64_t key_heads,
    float epsilon) {
  constexpr int kHalfDim = 64;
  constexpr int kHeadDim = 128;
  constexpr int kBlockSize = kHalfDim;
  const int64_t token = blockIdx.y;
  const int64_t combined_head = blockIdx.x;
  const bool is_query = combined_head < query_heads;
  const int64_t head = is_query ? combined_head : combined_head - query_heads;
  scalar_t* tensor = is_query ? query : key;
  const scalar_t* weight = is_query ? query_weight : key_weight;
  const int64_t heads = is_query ? query_heads : key_heads;
  const int64_t base = (token * heads + head) * kHeadDim;
  const int lane = threadIdx.x;

  const float first = to_float(tensor[base + lane]);
  const float second = to_float(tensor[base + lane + kHalfDim]);
  const float sum_sq = first * first + second * second;
  const float inv_rms = rsqrtf(
      block_reduce_sum<kBlockSize>(sum_sq) / static_cast<float>(kHeadDim) + epsilon);

  // Match the existing Qwen3 path: RMSNorm is rounded to the activation dtype
  // before RoPE performs its arithmetic in FP32.
  const float norm_first = to_float(from_float<scalar_t>(
      first * inv_rms * to_float(weight[lane])));
  const float norm_second = to_float(from_float<scalar_t>(
      second * inv_rms * to_float(weight[lane + kHalfDim])));
  const int64_t cache_base = positions[token] * kHeadDim;
  const float cosine = cos_sin_cache[cache_base + lane];
  const float sine = cos_sin_cache[cache_base + lane + kHalfDim];

  tensor[base + lane] = from_float<scalar_t>(
      norm_first * cosine - norm_second * sine);
  tensor[base + lane + kHalfDim] = from_float<scalar_t>(
      norm_second * cosine + norm_first * sine);
}

// V2: one warp owns one head. Four values/thread cover head_dim=128, so the
// reduction stays entirely in registers and warp shuffle instructions.
template <typename scalar_t>
__global__ void fused_qk_rms_norm_rope_h128_warp32_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ query_weight,
    const scalar_t* __restrict__ key_weight,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    int64_t query_heads,
    int64_t key_heads,
    float epsilon) {
  constexpr int kWarp = 32;
  constexpr int kHalfDim = 64;
  constexpr int kHeadDim = 128;
  const int64_t token = blockIdx.y;
  const int64_t combined_head = blockIdx.x;
  const bool is_query = combined_head < query_heads;
  const int64_t head = is_query ? combined_head : combined_head - query_heads;
  scalar_t* tensor = is_query ? query : key;
  const scalar_t* weight = is_query ? query_weight : key_weight;
  const int64_t heads = is_query ? query_heads : key_heads;
  const int64_t base = (token * heads + head) * kHeadDim;
  const int lane = threadIdx.x;
  const int first_index = lane;
  const int second_index = lane + kWarp;
  const int third_index = lane + kHalfDim;
  const int fourth_index = lane + kHalfDim + kWarp;

  const float first = to_float(tensor[base + first_index]);
  const float second = to_float(tensor[base + second_index]);
  const float third = to_float(tensor[base + third_index]);
  const float fourth = to_float(tensor[base + fourth_index]);
  float sum_sq = first * first + second * second + third * third + fourth * fourth;
  sum_sq = warp_reduce_sum(sum_sq);
  sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(kHeadDim) + epsilon);

  const float norm_first = to_float(from_float<scalar_t>(
      first * inv_rms * to_float(weight[first_index])));
  const float norm_second = to_float(from_float<scalar_t>(
      second * inv_rms * to_float(weight[second_index])));
  const float norm_third = to_float(from_float<scalar_t>(
      third * inv_rms * to_float(weight[third_index])));
  const float norm_fourth = to_float(from_float<scalar_t>(
      fourth * inv_rms * to_float(weight[fourth_index])));
  const int64_t cache_base = positions[token] * kHeadDim;
  const float cosine_first = cos_sin_cache[cache_base + first_index];
  const float cosine_second = cos_sin_cache[cache_base + second_index];
  const float sine_first = cos_sin_cache[cache_base + third_index];
  const float sine_second = cos_sin_cache[cache_base + fourth_index];

  tensor[base + first_index] = from_float<scalar_t>(
      norm_first * cosine_first - norm_third * sine_first);
  tensor[base + third_index] = from_float<scalar_t>(
      norm_third * cosine_first + norm_first * sine_first);
  tensor[base + second_index] = from_float<scalar_t>(
      norm_second * cosine_second - norm_fourth * sine_second);
  tensor[base + fourth_index] = from_float<scalar_t>(
      norm_fourth * cosine_second + norm_second * sine_second);
}

// V3 production path: extend V2's register-resident Q/K transform across the
// adjacent KV-cache boundary. K heads publish their rotated values and copy V
// directly into paged cache slots, eliminating a separate KV append launch and
// the extra global read of K.
template <typename scalar_t, int kWarpsPerBlock>
__global__ void fused_qk_rms_norm_rope_kv_h128_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ query_weight,
    const scalar_t* __restrict__ key_weight,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int32_t* __restrict__ slot_mapping,
    int64_t query_heads,
    int64_t key_heads,
    float epsilon) {
  constexpr int kWarp = 32;
  constexpr int kHalfDim = 64;
  constexpr int kHeadDim = 128;
  const int64_t token = blockIdx.y;
  const int warp = threadIdx.x / kWarp;
  const int64_t combined_head = blockIdx.x * kWarpsPerBlock + warp;
  if (combined_head >= query_heads + key_heads) {
    return;
  }
  const bool is_query = combined_head < query_heads;
  const int64_t head = is_query ? combined_head : combined_head - query_heads;
  scalar_t* tensor = is_query ? query : key;
  const scalar_t* weight = is_query ? query_weight : key_weight;
  const int64_t heads = is_query ? query_heads : key_heads;
  const int64_t base = (token * heads + head) * kHeadDim;
  const int lane = threadIdx.x % kWarp;
  const int indices[4] = {lane, lane + kWarp, lane + kHalfDim,
                          lane + kHalfDim + kWarp};

  const float first = to_float(tensor[base + indices[0]]);
  const float second = to_float(tensor[base + indices[1]]);
  const float third = to_float(tensor[base + indices[2]]);
  const float fourth = to_float(tensor[base + indices[3]]);
  float sum_sq = first * first + second * second + third * third + fourth * fourth;
  sum_sq = warp_reduce_sum(sum_sq);
  sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(kHeadDim) + epsilon);

  const float norm_first = to_float(from_float<scalar_t>(
      first * inv_rms * to_float(weight[indices[0]])));
  const float norm_second = to_float(from_float<scalar_t>(
      second * inv_rms * to_float(weight[indices[1]])));
  const float norm_third = to_float(from_float<scalar_t>(
      third * inv_rms * to_float(weight[indices[2]])));
  const float norm_fourth = to_float(from_float<scalar_t>(
      fourth * inv_rms * to_float(weight[indices[3]])));
  const int64_t rope_base = positions[token] * kHeadDim;
  const float cosine_first = cos_sin_cache[rope_base + indices[0]];
  const float cosine_second = cos_sin_cache[rope_base + indices[1]];
  const float sine_first = cos_sin_cache[rope_base + indices[2]];
  const float sine_second = cos_sin_cache[rope_base + indices[3]];
  const scalar_t outputs[4] = {
      from_float<scalar_t>(norm_first * cosine_first - norm_third * sine_first),
      from_float<scalar_t>(norm_second * cosine_second - norm_fourth * sine_second),
      from_float<scalar_t>(norm_third * cosine_first + norm_first * sine_first),
      from_float<scalar_t>(norm_fourth * cosine_second + norm_second * sine_second)};

#pragma unroll
  for (int item = 0; item < 4; ++item) {
    tensor[base + indices[item]] = outputs[item];
  }

  if (!is_query) {
    const int32_t slot = slot_mapping[token];
    if (slot >= 0) {
      const int64_t cache_base =
          (static_cast<int64_t>(slot) * key_heads + head) * kHeadDim;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        key_cache[cache_base + indices[item]] = outputs[item];
        value_cache[cache_base + indices[item]] = value[base + indices[item]];
      }
    }
  }
}

void check_common(const torch::Tensor& input, const torch::Tensor& weight) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
  TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  TORCH_CHECK(input.size(-1) == weight.numel(), "weight size must equal input hidden size");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight dtypes must match");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16 ||
          input.scalar_type() == torch::kFloat32,
      "supported dtypes are float16, bfloat16, and float32");
}

}  // namespace

void rms_norm_out_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    double epsilon,
    int64_t version);

void fused_add_rms_norm_out_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor residual_output,
    double epsilon,
    int64_t version);

torch::Tensor rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double epsilon,
    int64_t version) {
  auto output = torch::empty_like(input);
  rms_norm_out_cuda(input, weight, output, epsilon, version);
  return output;
}

void rms_norm_out_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    double epsilon,
    int64_t version) {
  check_common(input, weight);
  TORCH_CHECK(version == 1 || version == 2, "RMSNorm version must be 1 or 2");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "output must be contiguous CUDA");
  TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input");
  c10::cuda::CUDAGuard device_guard(input.device());

  const int64_t hidden_size = input.size(-1);
  const int64_t rows = input.numel() / hidden_size;
  constexpr int kBlockSize = 256;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "rms_norm_cuda",
      [&] {
        if (version == 1) {
          rms_norm_shared_kernel<scalar_t, kBlockSize><<<
              rows, kBlockSize, kBlockSize * sizeof(float), stream>>>(
              input.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              hidden_size,
              static_cast<float>(epsilon));
        } else {
          rms_norm_warp_kernel<scalar_t, kBlockSize><<<rows, kBlockSize, 0, stream>>>(
              input.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              hidden_size,
              static_cast<float>(epsilon));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_add_rms_norm_inplace_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double epsilon,
    int64_t version) {
  // Match the vLLM/FlashInfer contract: residual receives input + residual,
  // and input receives the normalized result. Every kernel version completes
  // its input reads before writing input, so these aliases are safe.
  fused_add_rms_norm_out_cuda(
      input, residual, weight, input, residual, epsilon, version);
}

void fused_qk_rms_norm_rope_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor query_weight,
    torch::Tensor key_weight,
    torch::Tensor positions,
    torch::Tensor cos_sin_cache,
    double epsilon,
    int64_t version) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda(), "query and key must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous(),
              "query and key must be contiguous");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3,
              "query and key must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(query.size(0) == key.size(0), "query and key token counts must match");
  TORCH_CHECK(query.size(2) == 128 && key.size(2) == 128,
              "fused QK Norm+RoPE requires head_dim=128");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key dtypes must match");
  TORCH_CHECK(query_weight.is_cuda() && key_weight.is_cuda(),
              "Q/K weights must be CUDA tensors");
  TORCH_CHECK(query_weight.is_contiguous() && key_weight.is_contiguous(),
              "Q/K weights must be contiguous");
  TORCH_CHECK(query_weight.numel() == 128 && key_weight.numel() == 128,
              "Q/K weights must contain 128 elements");
  TORCH_CHECK(query_weight.scalar_type() == query.scalar_type() &&
                  key_weight.scalar_type() == query.scalar_type(),
              "Q/K weight dtypes must match activations");
  TORCH_CHECK(positions.is_cuda() && positions.is_contiguous(),
              "positions must be a contiguous CUDA tensor");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64,
              "positions must use int64 dtype");
  TORCH_CHECK(positions.numel() == query.size(0),
              "positions length must match token count");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.is_contiguous(),
              "cos/sin cache must be a contiguous CUDA tensor");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32,
              "cos/sin cache must use float32 dtype");
  TORCH_CHECK(cos_sin_cache.size(-1) == 128,
              "cos/sin cache last dimension must be 128");
  TORCH_CHECK(version == 1 || version == 2,
              "QK Norm+RoPE version must be 1 or 2");
  TORCH_CHECK(query.device() == key.device() && query.device() == query_weight.device() &&
                  query.device() == key_weight.device() && query.device() == positions.device() &&
                  query.device() == cos_sin_cache.device(),
              "all tensors must be on the same CUDA device");
  c10::cuda::CUDAGuard device_guard(query.device());

  const dim3 grid(query.size(1) + key.size(1), query.size(0));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "fused_qk_rms_norm_rope_cuda",
      [&] {
        if (version == 1) {
          fused_qk_rms_norm_rope_h128_block64_kernel<scalar_t><<<grid, 64, 0, stream>>>(
              query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
              query_weight.data_ptr<scalar_t>(), key_weight.data_ptr<scalar_t>(),
              positions.data_ptr<int64_t>(), cos_sin_cache.data_ptr<float>(),
              query.size(1), key.size(1), static_cast<float>(epsilon));
        } else {
          fused_qk_rms_norm_rope_h128_warp32_kernel<scalar_t><<<grid, 32, 0, stream>>>(
              query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
              query_weight.data_ptr<scalar_t>(), key_weight.data_ptr<scalar_t>(),
              positions.data_ptr<int64_t>(), cos_sin_cache.data_ptr<float>(),
              query.size(1), key.size(1), static_cast<float>(epsilon));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_qk_rms_norm_rope_kv_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_weight,
    torch::Tensor key_weight,
    torch::Tensor positions,
    torch::Tensor cos_sin_cache,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping,
    double epsilon,
    int64_t version) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "Q/K/V must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "Q/K/V must be contiguous");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "Q/K/V must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(query.size(0) == key.size(0) && key.sizes() == value.sizes(),
              "Q/K/V token and KV shapes must match");
  TORCH_CHECK(query.size(2) == 128 && key.size(2) == 128,
              "fused QK Norm+RoPE+KV requires head_dim=128");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  key.scalar_type() == value.scalar_type(),
              "Q/K/V dtypes must match");
  TORCH_CHECK(query_weight.is_cuda() && key_weight.is_cuda() &&
                  query_weight.is_contiguous() && key_weight.is_contiguous(),
              "Q/K weights must be contiguous CUDA tensors");
  TORCH_CHECK(query_weight.numel() == 128 && key_weight.numel() == 128,
              "Q/K weights must contain 128 elements");
  TORCH_CHECK(query_weight.scalar_type() == query.scalar_type() &&
                  key_weight.scalar_type() == query.scalar_type(),
              "Q/K weight dtypes must match activations");
  TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() &&
                  positions.scalar_type() == torch::kInt64 &&
                  positions.numel() == query.size(0),
              "positions must be contiguous CUDA int64 with one entry per token");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.is_contiguous() &&
                  cos_sin_cache.scalar_type() == torch::kFloat32 &&
                  cos_sin_cache.size(-1) == 128,
              "cos/sin cache must be contiguous CUDA float32 with width 128");
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda() &&
                  key_cache.is_contiguous() && value_cache.is_contiguous(),
              "KV caches must be contiguous CUDA tensors");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes() &&
                  key_cache.scalar_type() == query.scalar_type() &&
                  value_cache.scalar_type() == query.scalar_type(),
              "KV cache shapes and dtypes must match activations");
  TORCH_CHECK(key_cache.size(-2) == key.size(1) && key_cache.size(-1) == 128,
              "KV cache must end in [key_heads, 128]");
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.is_contiguous() &&
                  slot_mapping.scalar_type() == torch::kInt &&
                  slot_mapping.numel() == query.size(0),
              "slot_mapping must be contiguous CUDA int32 with one entry per token");
  TORCH_CHECK(version == 1 || version == 4,
              "QK Norm+RoPE+KV version must be 1 or 4 warps per block");
  TORCH_CHECK(query.device() == key.device() && query.device() == value.device() &&
                  query.device() == query_weight.device() &&
                  query.device() == key_weight.device() &&
                  query.device() == positions.device() &&
                  query.device() == cos_sin_cache.device() &&
                  query.device() == key_cache.device() &&
                  query.device() == value_cache.device() &&
                  query.device() == slot_mapping.device(),
              "all tensors must be on the same CUDA device");
  c10::cuda::CUDAGuard device_guard(query.device());

  const int64_t warps_per_block = version;
  const dim3 grid(
      (query.size(1) + key.size(1) + warps_per_block - 1) / warps_per_block,
      query.size(0));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "fused_qk_rms_norm_rope_kv_cuda",
      [&] {
        if (version == 1) {
          fused_qk_rms_norm_rope_kv_h128_kernel<scalar_t, 1><<<grid, 32, 0, stream>>>(
              query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(), query_weight.data_ptr<scalar_t>(),
              key_weight.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
              cos_sin_cache.data_ptr<float>(), key_cache.data_ptr<scalar_t>(),
              value_cache.data_ptr<scalar_t>(), slot_mapping.data_ptr<int32_t>(),
              query.size(1), key.size(1), static_cast<float>(epsilon));
        } else {
          fused_qk_rms_norm_rope_kv_h128_kernel<scalar_t, 4><<<grid, 128, 0, stream>>>(
              query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(), query_weight.data_ptr<scalar_t>(),
              key_weight.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
              cos_sin_cache.data_ptr<float>(), key_cache.data_ptr<scalar_t>(),
              value_cache.data_ptr<scalar_t>(), slot_mapping.data_ptr<int32_t>(),
              query.size(1), key.size(1), static_cast<float>(epsilon));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> fused_add_rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double epsilon,
    int64_t version) {
  auto output = torch::empty_like(input);
  auto residual_output = torch::empty_like(input);
  fused_add_rms_norm_out_cuda(
      input, residual, weight, output, residual_output, epsilon, version);
  return {output, residual_output};
}

void fused_add_rms_norm_out_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor residual_output,
    double epsilon,
    int64_t version) {
  check_common(input, weight);
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous(), "residual must be contiguous CUDA");
  TORCH_CHECK(residual.sizes() == input.sizes(), "residual shape must match input");
  TORCH_CHECK(residual.scalar_type() == input.scalar_type(), "residual dtype must match input");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "output must be contiguous CUDA");
  TORCH_CHECK(residual_output.is_cuda() && residual_output.is_contiguous(),
              "residual_output must be contiguous CUDA");
  TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
  TORCH_CHECK(residual_output.sizes() == input.sizes(), "residual_output shape must match input");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input");
  TORCH_CHECK(residual_output.scalar_type() == input.scalar_type(),
              "residual_output dtype must match input");
  TORCH_CHECK(version >= 0 && version <= 4 && version != 1,
              "fused version must be 0 (auto), 2, 3, or 4");
  c10::cuda::CUDAGuard device_guard(input.device());

  const int64_t hidden_size = input.size(-1);
  const int64_t rows = input.numel() / hidden_size;
  constexpr int kBlockSize = 256;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int64_t selected_version = version;
  if (selected_version == 0) {
    if (hidden_size != 1024) {
      selected_version = 2;
    } else if (rows > 32 && rows <= 192) {
      selected_version = 3;
    } else {
      selected_version = 4;
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fused_add_rms_norm_cuda",
      [&] {
        if (selected_version == 3) {
          TORCH_CHECK(hidden_size == 1024, "V3 is specialized for hidden_size=1024");
          fused_add_rms_norm_h1024_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
              input.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              residual_output.data_ptr<scalar_t>(),
              static_cast<float>(epsilon));
        } else if (selected_version == 4) {
          TORCH_CHECK(hidden_size == 1024, "V4 is specialized for hidden_size=1024");
          constexpr int kPackedBlockSize = 1024 / Pack128<scalar_t>::kItems;
          fused_add_rms_norm_h1024_packed_kernel<scalar_t>
              <<<rows, kPackedBlockSize, 0, stream>>>(
                  input.data_ptr<scalar_t>(),
                  residual.data_ptr<scalar_t>(),
                  weight.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>(),
                  residual_output.data_ptr<scalar_t>(),
                  static_cast<float>(epsilon));
        } else {
          fused_add_rms_norm_warp_kernel<scalar_t, kBlockSize><<<rows, kBlockSize, 0, stream>>>(
              input.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              residual_output.data_ptr<scalar_t>(),
              hidden_size,
              static_cast<float>(epsilon));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
