import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

setup(
    name="sm120-fused-rmsnorm",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="sm120_rmsnorm._C",
            sources=["csrc/bindings.cpp", "csrc/rmsnorm_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)

