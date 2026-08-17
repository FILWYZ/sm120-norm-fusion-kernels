# SM120 LLM Norm 融合算子优化

面向 RTX 5060 Laptop（Blackwell SM120）的 CUDA C++ 性能工程项目，包含
RMSNorm、Residual Add + RMSNorm，以及 Qwen3 Decode 热路径中的 Q/K
RMSNorm + RoPE + Paged KV Store 融合。

项目按照“基线实现 → Warp/访存优化 → NCU/SASS 验证 → 工业算子对比 →
真实推理引擎集成”的方式推进。重点不是证明自定义 CUDA 总能胜过框架，而是
展示如何确定融合边界、解释硬件指标，并在端到端无收益时收窄调度范围。

关联推理引擎：
[FILWYZ/nano-vllm-sm120-optimized](https://github.com/FILWYZ/nano-vllm-sm120-optimized)。

## 核心成果

| 方向 | 结果 |
|---|---|
| PyTorch Native 基线 | FP16/BF16 `hidden_size=1024` 下，预分配 RMSNorm 快 2.65–3.40× |
| FlashInfer 0.6.6 公平对比 | 相同原地语义下，FP16 延迟降低 2.9–13.2%，BF16 降低 1.9–13.2% |
| Q/K Norm+RoPE 微基准 | 相对等价 `torch.compile` 链路快 4.9–8.1× |
| nano-vLLM 端到端 | batch=1 Decode +1.77% / +2.31%，输出吞吐 +1.24% / +1.72% |
| 工程验证 | 180 项 GPU 测试；memcheck 0 errors；racecheck 0 hazards |

所有性能结论只适用于文档记录的 GPU、dtype、Shape、软件版本和测试协议。

## 已实现版本

| 版本 | 实现 | 优化目标 |
|---|---|---|
| V1 | 标量访存 + Shared Memory 树形归约 | 可读、可验证的 CUDA 基线 |
| V2 | 分层 Warp Shuffle 归约 | 减少同步和 Shared Memory 流量 |
| V2 fused | Residual Add + 通用 RMSNorm | 删除一次 Kernel Launch |
| V3 fused | 1024 宽度寄存器缓存 | 归约前后复用每线程四个元素 |
| V4 fused | dtype-aware 128-bit Pack | 减少全局访存指令数量 |
| Auto | SM120 实测 Shape Dispatch | 按行数选择 V4/V3/V4，其他宽度回退 V2 |

Residual Add + RMSNorm 提供与 vLLM/FlashInfer 风格一致的原地接口：

```text
residual = residual + input
input = RMSNorm(residual) * weight
```

## Qwen3 Decode 融合

生产快速路径跨越真实的算子边界，一次完成：

1. Q/K 逐 Head RMSNorm，FP32 累加；
2. 激活 dtype 舍入后的 FP32 半分式 RoPE；
3. 旋转后 K 散写到 16-token Paged KV Cache；
4. V 写入同一 Cache Slot。

一个 warp 负责一个 128 元素 Head，每个线程把四个元素保存在寄存器中。
项目还保留 four-warps/block 消融版本：该版本达到 100% 理论 Occupancy，
独立 Kernel 略快，但没有带来稳定端到端收益。

因此 nano-vLLM 最终只在 **SM120 + Qwen3 head_dim=128 + batch=1 Decode**
启用 one-warp/head 融合，Prefill 和宽 Batch 回退 PyTorch Inductor。

详细结果见
[`QK_NORM_ROPE_KV_RESULTS.md`](docs/QK_NORM_ROPE_KV_RESULTS.md)。

## SASS 与 NCU 证据

- V4 FP16/BF16 SASS 包含 `LDG.E.128` 和 `STG.E.128`，证明 16-byte Pack
  确实生成 128-bit 全局访存。
- Q/K Kernel SASS 包含 `SHFL.DOWN`、`SHFL.IDX` 和 `MUFU.RSQ`，生产版本
  不包含 Block Barrier。
- NCU 显示 V4 虽提高显存吞吐但增加寄存器使用，说明向量化、Occupancy
  和 Shape 之间需要实测权衡。

```bash
bash scripts/check_sass.sh
bash scripts/profile_ncu.sh
```

## 构建与测试

要求：WSL 可见 NVIDIA 驱动，安装与 PyTorch 版本匹配的 CUDA Toolkit、
NVCC、Ninja 和 C++ 编译器。

```bash
python -m pip install -U pip ninja pytest

CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=12.0 \
python -m pip install -e . --no-build-isolation

pytest -q
```

环境检查：

```bash
nvidia-smi
nvcc --version

python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY
```

## Benchmark

RMSNorm：

```bash
python -m benchmarks.benchmark \
  --tokens 1 4 16 64 256 \
  --hidden 768 1024 2048 4096 \
  --dtype fp16 \
  --output benchmark_results/fp16.json
```

Q/K Norm+RoPE：

```bash
python -m benchmarks.qk_norm_rope \
  --dtype bf16 \
  --output benchmark_results/qk_norm_rope_bf16.json

python -m benchmarks.profile_qk_norm_rope \
  --tokens 1 --version 1 --store-kv
```

文档索引：

- [`FIRST_RESULTS.md`](docs/FIRST_RESULTS.md)：PyTorch Native 基线与 NCU 结果。
- [`INDUSTRIAL_BASELINE.md`](docs/INDUSTRIAL_BASELINE.md)：FlashInfer 公平对比。
- [`QK_NORM_ROPE_KV_RESULTS.md`](docs/QK_NORM_ROPE_KV_RESULTS.md)：Qwen3 融合与端到端调度。

## 正确性契约

- 输入、Residual 和 Weight 位于同一 CUDA Device 且 dtype 匹配；
- 支持 FP16、BF16、FP32，归约使用 FP32；
- 通用 V1/V2 支持非整数倍 Hidden Size；
- V3/V4 只支持 `hidden_size=1024`，其他宽度由 Auto 回退 V2；
- Q/K 融合要求 `head_dim=128`、连续布局、FP32 RoPE Cache、int64 Positions
  和 int32 Slot Mapping；
- 不支持的引擎 Shape 不强行启用自定义 Kernel。

## 结论边界

本仓库证明的是一套完整的 CUDA 性能工程流程，而不是“自定义 Kernel 必然
优于 PyTorch、FlashInfer 或 vLLM”。独立算子收益必须经过真实推理引擎 A/B
验证；无法稳定获益的 Shape 应保留框架实现。
