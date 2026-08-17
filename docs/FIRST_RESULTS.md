# SM120 首轮 RMSNorm 优化结果

## 测试环境与协议

- GPU：NVIDIA GeForce RTX 5060 Laptop，Compute Capability 12.0，8 GiB；
- Driver：610.74；CUDA Toolkit：12.8；
- PyTorch：2.11.0+cu128；归约精度：FP32；
- Shape：`hidden_size=1024`，tokens `{1, 16, 64, 256}`；
- 五个独立 Python 进程；每进程 100 次预热、500 次迭代、三个 CUDA Event
  样本；表格取五个进程中位数的中位数。

PyTorch Native 基线为独立 Residual Add 后调用
`torch.nn.functional.rms_norm`。自定义 CUDA 使用预分配输出，避免把输出
分配开销混入 Kernel 对比。

## FP16 延迟

| Tokens | PyTorch Native | CUDA V2 通用 | CUDA V3 寄存器缓存 | CUDA V4 Pack | 最佳加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.525 μs | 6.724 μs | 5.628 μs | **5.425 μs** | **3.23×** |
| 16 | 17.256 μs | 6.284 μs | 5.518 μs | **5.077 μs** | **3.40×** |
| 64 | 17.112 μs | 6.505 μs | 6.132 μs | **5.900 μs** | **2.90×** |
| 256 | 16.821 μs | 7.528 μs | **6.357 μs** | 6.377 μs | **2.65×** |

## BF16 延迟

| Tokens | PyTorch Native | CUDA V2 通用 | CUDA V3 寄存器缓存 | CUDA V4 Pack | 最佳加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.265 μs | 6.335 μs | 5.402 μs | **5.277 μs** | **3.27×** |
| 16 | 17.234 μs | 6.178 μs | 5.439 μs | **5.315 μs** | **3.24×** |
| 64 | 17.183 μs | 6.443 μs | 5.905 μs | **5.821 μs** | **2.95×** |
| 256 | 16.965 μs | 7.611 μs | 6.445 μs | **6.363 μs** | **2.67×** |

## NCU：FP16，tokens=256

| 指标 | V3 寄存器缓存 | V4 128-bit Pack |
|---|---:|---:|
| Kernel 时间 | 5.73 μs | **5.12 μs** |
| Memory Throughput | 184.31 GB/s | **206.25 GB/s** |
| DRAM Throughput | 42.12% | **47.39%** |
| Registers/Thread | **20** | 28 |
| Achieved Occupancy | **74.65%** | 51.72% |

V4 的纯 Kernel 时间降低约 10.6%，显存吞吐提高约 11.9%。128-bit Pack 增加
寄存器使用并降低 Occupancy，但减少访存指令后仍取得更短 Kernel 时间。
跨进程中位数中，FP16 tokens=256 的 V3 略快，说明最终 Dispatcher 应根据
Shape 实测选择，而不是所有情况固定使用 V4。

一次 FP16 进程出现系统级延迟抖动，同时影响 PyTorch 和自定义实现。原始数据
保留该进程，最终使用跨进程中位数，不挑选最快运行。

## 正确性与复现

首轮阶段共有 `59 passed`，覆盖 FP16/BF16/FP32；V1/V2 包含非整数倍 Hidden
Size，V3/V4 强制 `hidden_size=1024`。当前最终仓库测试数已扩展到 180。

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=12.0 \
python -m pip install -e . --no-build-isolation

pytest -q

python -m benchmarks.run_repeated \
  --dtype fp16 --repeats 5 --warmup 100 --iterations 500
```

以上是单机结果。引用时必须同时保留 GPU、dtype、Shape、基线和独立进程协议。
