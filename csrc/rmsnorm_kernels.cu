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
