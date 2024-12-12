# ROCm HIP runtime

Runtime that runs on ROCm HIP supported AMD GPUs.

Matrix multiplication acceleration is based on [rocwmma][] by default. Note that kernel compilation time
with [rocwmma][] might be slow.

For RDNA3 GPUs, a dedicated compiler using [WMMA intrinsics][] is available with the feature `wmma-intrinsics`.
It offers much faster kernel compilation time and better performances on some kernels. Feel free to benchmark
with your use cases.

[rocwmma]: https://github.com/ROCm/rocWMMA
[WMMA intrinsics]: https://gpuopen.com/learn/wmma_on_rdna3/
