# Qwen3 Q/K Norm + RoPE + Paged KV Store 融合结果

## 算子契约

Qwen3-0.6B 的目标 Shape 为 Q `[T, 16, 128]`、K/V `[T, 8, 128]`，激活和
Weight 使用 BF16，RoPE Cache 使用 FP32，Positions 使用 int64，Paged KV
Slot Mapping 使用 int32。

Kernel 保持原路径的数值顺序：FP32 RMS Reduction → RMSNorm 后舍入到激活
dtype → FP32 半分式 RoPE → 再次舍入到激活 dtype。

Decode 算子原地修改 Q/K，并将旋转后的 K 和 V 散写到
`[blocks, page_size, 8, 128]` Cache。PyTorch Custom Operator Schema 显式
声明所有 Mutation，FakeTensor 实现支持 `torch.compile(fullgraph=True)`
和 CUDA Graph Capture。

## 微基准

环境：RTX 5060 Laptop SM120、PyTorch 2.11.0+cu128、CUDA Toolkit 12.8、
BF16、100 次预热、1,000 次迭代、三个 CUDA Event 样本。

| Tokens | `torch.compile` | 融合 Q/K Norm+RoPE | 加速比 |
|---:|---:|---:|---:|
| 1 | 46.34 μs | 9.36 μs | 4.95× |
| 4 | 55.61 μs | 7.05 μs | 7.89× |
| 16 | 53.32 μs | 6.62 μs | 8.06× |
| 64 | 54.65 μs | 7.11 μs | 7.69× |
| 256 | 54.17 μs | 11.03 μs | 4.91× |

该表只隔离 Q/K Norm+RoPE；生产版本还会删除独立 Triton KV Append Launch
和一次 K 的全局内存回读。

## NCU 与 SASS

64 个 BF16 Token 时，寄存器化 one-warp Q/K Kernel 为 4.48 μs，双 warp
Shared Reduction 版本为 4.86 μs；256 Token 时分别为 10.40 μs 和
13.47 μs。

最终 SASS 包含 `SHFL.DOWN`、`SHFL.IDX`、`MUFU.RSQ` 和全局 Load/Store，
不包含 Block Barrier。

KV 融合消融比较 one-warp/block 与 four-warps/block：NCU 在 T=1/T=8 下
分别得到 2.82/2.94 μs 和 2.66/2.75 μs。four-warps 版本虽然独立 Kernel
略快且达到 100% 理论 Occupancy，但没有产生稳定端到端收益，因此生产
Dispatch 使用经过端到端验证的 one-warp 版本，两版代码均保留用于复现。

## 端到端结果

协议：Qwen3-0.6B BF16、FlashInfer Attention、16-token KV Page、CUDA Graph
Decode；每后端五个独立进程，A/B 顺序交替；每负载两次测量；最终取进程
均值的中位数。

| 请求数 | Prompt | Decode 变化 | 输出吞吐变化 |
|---:|---:|---:|---:|
| 1 | 64 | **+1.77%** | **+1.24%** |
| 1 | 256 | **+2.31%** | **+1.72%** |

Prefill 和 batch>1 Decode 使用现有 Inductor 路径。两组配置在这些 Shape
上的跨进程差异只视为测量噪声，不归因于自定义 Kernel。窄 Dispatch 避免了
全局启用时观察到的 Prefill 回退。

## 验证

- CUDA 正确性测试：180 项；
- nano-vLLM：41 项测试和 5 个 subtests；
- `torch.compile(fullgraph=True)` Custom Operator 测试通过；
- 真实推理引擎 CUDA Graph Capture/Replay 通过；
- Compute Sanitizer memcheck：0 errors；
- Compute Sanitizer racecheck：0 hazards；
- CUDA 13.3 NCU 与 Cubin 反汇编验证 CUDA 12.8 构建的 SM120 代码。

公开结论只适用于记录的模型、GPU、软件栈和负载，不是相对官方 vLLM、
FlashInfer、Inductor 或其他架构的通用性能结论。
