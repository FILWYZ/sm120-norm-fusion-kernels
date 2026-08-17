#include <torch/extension.h>

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
}
