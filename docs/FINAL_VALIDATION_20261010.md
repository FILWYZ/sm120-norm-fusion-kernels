# Final commit validation — 2026-10-10

Tested source commit: `d990cca2af201203ac810cb0c63c13049dabdd1c` (`docs: publish RTX 5060 kernel shape map`). Hardware: RTX 5060 Laptop, SM120, driver 610.74. Environment: PyTorch 2.14.1+cu130, CUDA Runtime 13.0, CUDA Toolkit/NVCC 13.3, FlashInfer 0.6.6, Triton 3.8.0.

The extension was rebuilt from this clean checkout for SM120 and installed into the test environment. Full GPU suite result: **180 passed**.

Focused Compute Sanitizer checks covered fused RMSNorm, Q/K Norm + RoPE, and Q/K Norm + RoPE + KV store:

- memcheck: 3 selected tests passed, **0 errors**;
- racecheck: 3 selected tests passed, **0 hazards**.

The sanitizer run was intentionally limited to representative kernel paths; the entire 180-test suite was not run under sanitizer. The documentation commit did not change CUDA kernel source or runtime code.
