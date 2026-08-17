# FlashInfer fused add RMSNorm 公平基线

## 对比范围

对比使用 FlashInfer 和 vLLM 风格的相同原地语义：

```text
residual = residual + input
input = RMSNorm(residual) * weight
```

环境：RTX 5060 Laptop SM120、PyTorch 2.11.0+cu128、FlashInfer 0.6.6、
CUDA Toolkit 12.8、`hidden_size=1024`。

为避免重复原地调用时数值不断累积，同时不把 Reset Kernel 加入计时区域，
延迟基准在随机分配 Buffer 后使用零 Input 和零 Weight。两种实现仍执行相同的
Load、Reduction、Normalization 和 Store 路径；随机输入和 Weight 的正确性
由独立 FP16/BF16/FP32 测试覆盖。

测试协议：

- 五个独立 Python 进程；
- 每进程三个 CUDA Event 样本；
- 每样本 100 次预热、500 次正式迭代；
- 表格取五个进程中位数的中位数；
- 自定义实现和 FlashInfer 均使用预分配、原地接口。

## FP16 结果

| Tokens | Custom Auto | FlashInfer 0.6.6 | 加速比 | 延迟降低 |
|---:|---:|---:|---:|---:|
| 1 | 5.229 μs | 6.027 μs | 1.152× | 13.23% |
| 16 | 5.295 μs | 5.963 μs | 1.126× | 11.20% |
| 64 | 5.831 μs | 6.083 μs | 1.043× | 4.15% |
| 256 | 6.241 μs | 6.428 μs | 1.030× | 2.92% |

## BF16 结果

| Tokens | Custom Auto | FlashInfer 0.6.6 | 加速比 | 延迟降低 |
|---:|---:|---:|---:|---:|
| 1 | 5.280 μs | 6.085 μs | 1.152× | 13.23% |
| 16 | 5.236 μs | 6.035 μs | 1.153× | 13.24% |
| 64 | 5.815 μs | 6.136 μs | 1.055× | 5.23% |
| 256 | 6.274 μs | 6.396 μs | 1.019× | 1.91% |

原始分进程数据和汇总 JSON 位于
`benchmark_results/flashinfer_repeated/`。

## SM120 Dispatch 规则

| Rows | 选择的 Kernel |
|---:|---|
| 1–32 | V4 128-bit Pack |
| 33–192 | V3 寄存器缓存 |
| 193+ | V4 128-bit Pack |
| 非 1024 Hidden Size | V2 通用回退 |

该阈值来自单张 SM120 和单一 Hidden Size 的实测，不代表其他 GPU 的最优规则。

## SASS 证据

CUDA 13.3 `cuobjdump/nvdisasm` 显示，CUDA 12.8 构建的 V4 FP16/BF16
特化包含 `LDG.E.128` 和 `STG.E.128`；V3 使用标量 `LDG.E.U16` 和
`STG.E.U16`。这证明源码中的 `alignas(16)` Pack 确实生成 128-bit
全局访存，而不是只停留在源码意图。

## 结论边界

安全结论是：在记录的 FP16/BF16 Shape 上，该 SM120 特化实现比
FlashInfer 0.6.6 对应算子更快。它不是对所有 vLLM、FlashInfer、Inductor、
GPU、dtype 或 Hidden Size 的通用结论；算子收益是否能传递到推理引擎，必须
继续通过端到端 A/B 验证。
