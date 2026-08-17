#include <torch/extension.h>
#include <torch/library.h>

#include <vector>

torch::Tensor rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double epsilon,
    int64_t version);

void rms_norm_out_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    double epsilon,
    int64_t version);

std::vector<torch::Tensor> fused_add_rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
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

void fused_add_rms_norm_inplace_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double epsilon,
    int64_t version);

void fused_qk_rms_norm_rope_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor query_weight,
    torch::Tensor key_weight,
    torch::Tensor positions,
    torch::Tensor cos_sin_cache,
    double epsilon,
    int64_t version);

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
    int64_t version);

TORCH_LIBRARY(sm120_norm, module) {
  module.def(
      "fused_qk_rms_norm_rope_(Tensor(a!) query, Tensor(b!) key, "
      "Tensor query_weight, Tensor key_weight, Tensor positions, "
      "Tensor cos_sin_cache, float epsilon=1e-6, int version=2) -> ()");
  module.def(
      "fused_qk_rms_norm_rope_kv_(Tensor(a!) query, Tensor(b!) key, Tensor value, "
      "Tensor query_weight, Tensor key_weight, Tensor positions, "
      "Tensor cos_sin_cache, Tensor(c!) key_cache, Tensor(d!) value_cache, "
      "Tensor slot_mapping, float epsilon=1e-6, int version=1) -> ()");
}

TORCH_LIBRARY_IMPL(sm120_norm, CUDA, module) {
  module.impl("fused_qk_rms_norm_rope_", &fused_qk_rms_norm_rope_cuda);
  module.impl("fused_qk_rms_norm_rope_kv_", &fused_qk_rms_norm_rope_kv_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rms_norm", &rms_norm_cuda, "RMSNorm forward (CUDA)");
  module.def("rms_norm_out", &rms_norm_out_cuda, "RMSNorm forward into preallocated output (CUDA)");
  module.def(
      "fused_add_rms_norm",
      &fused_add_rms_norm_cuda,
      "Fused residual add + RMSNorm forward (CUDA)");
  module.def(
      "fused_add_rms_norm_out",
      &fused_add_rms_norm_out_cuda,
      "Fused residual add + RMSNorm into preallocated outputs (CUDA)");
  module.def(
      "fused_add_rms_norm_inplace",
      &fused_add_rms_norm_inplace_cuda,
      "In-place fused residual add + RMSNorm (CUDA)");
  module.def(
      "fused_qk_rms_norm_rope",
      &fused_qk_rms_norm_rope_cuda,
      "In-place fused Q/K RMSNorm + RoPE (CUDA)");
  module.def(
      "fused_qk_rms_norm_rope_kv",
      &fused_qk_rms_norm_rope_kv_cuda,
      "In-place fused Q/K RMSNorm + RoPE + paged KV store (CUDA)");
}
