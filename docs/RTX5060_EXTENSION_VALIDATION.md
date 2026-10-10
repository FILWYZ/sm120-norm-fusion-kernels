# RTX 5060 扩展安装与真实负载验证

日期：2026-10-07

## 当前设备与实际加载链路

- GPU：NVIDIA GeForce RTX 5060，SM120 / Compute Capability 12.0，约 8 GiB。
- nano-vLLM Python：3.12；PyTorch 2.14.1+cu130（PyTorch CUDA runtime 13.0）。
- 扩展：CUDA Toolkit/NVCC 13.3，目标架构 sm_120。NVCC 与 PyTorch CUDA minor 版本不同，会有 warning；本次 wheel 构建、导入和 GPU 测试均成功。
- 扩展已用非 editable wheel 安装到 /home/asus/nano-vllm/.venv。实际加载文件：
  - /home/asus/nano-vllm/.venv/lib/python3.12/site-packages/sm120_rmsnorm/__init__.py
  - /home/asus/nano-vllm/.venv/lib/python3.12/site-packages/sm120_rmsnorm/_C.cpython-312-x86_64-linux-gnu.so
- nano-vLLM 的 auto 实际解析结果：RMSNorm=SM120，Q/K Norm+RoPE=SM120；GPU 可用。

重建安装命令：

    CUDA_HOME=/usr/local/cuda-13.3 TORCH_CUDA_ARCH_LIST=12.0 \
      ~/.local/bin/uv pip install --python /home/asus/nano-vllm/.venv/bin/python \
      --no-build-isolation --reinstall /home/asus/sm120-norm-fusion-kernels

FlashInfer 的既有 JIT cache 有过期的旧虚拟环境绝对路径。测试时使用独立 cache，未删除其他项目缓存：

    FLASHINFER_WORKSPACE_BASE=/home/asus/.cache/flashinfer_nano_vllm_cuda130 \
      /home/asus/nano-vllm/.venv/bin/python -m pytest -q

## 正确性验证

- SM120 kernel 仓库 GPU suite：180 passed。
- nano-vLLM 全量 suite：45 passed，5 个子测试通过。
- SM120 RMSNorm 与 Q/K Norm+RoPE 专项：18 passed。
- FlashInfer Attention 首次失败是旧 JIT cache 引用不存在的旧环境路径；换用独立 cache 后 5 个 Attention 测试通过，全量 suite 也通过。
- 多 token Q/K 融合试验期间，临时移除了 nano-vLLM 中的 batch=1 guard 并完成真实推理；测试后恢复原始生产源码。当前生产 Q/K 融合仍仅在 Decode batch=1 启用。

## RMSNorm 旧 Auto 与新 Auto：真实推理 A/B

模型 Qwen3-0.6B，FlashInfer Attention，Q/K Norm+RoPE 固定走 Torch。每个 workload 做 5 组独立进程配对，旧 Auto/新 Auto 顺序逐组交替。旧 Auto 精确重放原 dispatch（1024 hidden 且 33–192 rows 选 V3，其余走 Auto）；新 Auto 使用当前 V4 dispatch。表内是每组配对吞吐变化的中位数。

| Prompt / Decode | Batch | 输出吞吐变化 | Decode 吞吐变化 |
|---|---:|---:|---:|
| 64 / 32 | 8 | +1.15% | +0.72% |
| 64 / 32 | 16 | +0.28% | +0.72% |
| 64 / 32 | 32 | +0.01% | -0.32% |
| 64 / 32 | 33 | -0.22% | -0.11% |
| 64 / 32 | 48 | +0.41% | +0.15% |
| 64 / 32 | 64 | -0.01% | -0.18% |
| 256 / 16 | 8 | -0.76% | +0.58% |
| 256 / 16 | 32 | +0.97% | +0.40% |
| 256 / 16 | 33 | +0.05% | -0.12% |
| 256 / 16 | 48 | +0.14% | +0.42% |
| 256 / 16 | 64 | +0.32% | +0.58% |
| 64 / 64 | 8 | +0.90% | +0.26% |
| 64 / 64 | 32 | -0.03% | -0.17% |
| 64 / 64 | 33 | +0.75% | +0.67% |
| 64 / 64 | 48 | +0.03% | +0.04% |
| 64 / 64 | 64 | +0.19% | +0.05% |

新 dispatch 的额外收益没有在不同输入和 Decode 长度间稳定复现，大多处于 ±1%。已有 V3→V4 单算子交错基准仍支持 V4，但端到端证据不支持增加更细的 RMSNorm shape 分支。保留当前简洁规则，不把微基准加速宣称为模型吞吐提升。

## Q/K Norm + RoPE + KV：多 batch 试验

三组负载均做 5 组交错独立进程 A/B。为测试多行 kernel，试验进程临时开放了多 token Decode dispatch；试验后恢复原生产限制。每组 RMSNorm 固定用新 Auto，A/B 只切换 Q/K 后端。

| Prompt / Decode | Batch | SM120 相对 Torch 的输出吞吐变化 | Decode 吞吐变化 |
|---|---:|---:|---:|
| 64 / 32 | 8 | +1.36% | -0.11% |
| 64 / 32 | 16 | -0.10% | -0.23% |
| 64 / 32 | 32 | +0.40% | +0.28% |
| 64 / 32 | 64 | -0.41% | -0.37% |
| 256 / 16 | 1 | +2.77% | +3.16% |
| 256 / 16 | 8 | -0.40% | -1.29% |
| 256 / 16 | 32 | -0.28% | +0.09% |
| 256 / 16 | 64 | -0.40% | -0.36% |
| 64 / 64 | 1 | +2.78% | +2.61% |
| 64 / 64 | 8 | +0.07% | +0.10% |
| 64 / 64 | 32 | -0.30% | -0.21% |
| 64 / 64 | 64 | +0.07% | +0.20% |

batch=1 在两种 Decode 长度下都有约 2.8% 输出吞吐改善；batch 8–64 的差异基本在 ±0.5%，长 Prompt、batch 8 的 decode 有一次 -1.29%。所以生产只保留 batch=1 的 Q/K+RoPE+KV 融合；扩大到更大 batch 暂无端到端依据。

## NCU 硬件计数器

Nsight Compute 2026.2.1，full set（41 次 replay），tokens=64。计数器供资源画像使用；NCU 重放下的 kernel 时间不作为 A/B 性能结论。

| Kernel | GPU 时间样本 | 寄存器/线程 | 活跃 warp 比例 | DRAM 吞吐 | Local spill |
|---|---:|---:|---:|---:|---:|
| RMSNorm V3 | 2.944 μs | 19 | 33.24% | 91.48 GB/s | 0 |
| RMSNorm V4 | 3.040 μs | 28 | 15.48% | 88.67 GB/s | 0 |
| Q/K Norm+RoPE+KV | 5.088 μs | 29 | 37.66% | 110.69 GB/s | 0 |

V4 的寄存器增加、活跃 warp 比例下降，但没有 spill；这是其 128-bit 向量化与资源占用之间的权衡。V4 是否保留由交错 CUDA Event 微基准和 E2E A/B 决定。当前 NCU 结果没有指出可通过简单增大 warp 数获益的热点；下一轮内核改动应先针对特定 shape 做线程/向量宽度消融，再要求稳定 E2E 提升。

## 原始结果

- RMSNorm A/B：benchmark_results/rtx5060_local/rmsnorm_auto_matrix/
- Q/K 融合 A/B：benchmark_results/rtx5060_local/qk_batch_matrix/
- NCU .ncu-rep 和原始 CSV：benchmark_results/rtx5060_local/ncu/
- 可复现脚本：benchmarks/e2e_rmsnorm_paired.py、benchmarks/e2e_qk_paired.py、benchmarks/e2e_batch_dispatch.py
