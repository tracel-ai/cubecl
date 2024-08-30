# Cuda runtime

The runtime uses the lower level primitives from [cudarc](https://github.com/coreylowman/cudarc) to compile generated CUDA code into a ptx and execute it at runtime.

## Setup

By default, this runtime will attempt to detect the appropriate CUDA version during build time. To specify a fixed CUDA version, you can add `cudarc` to your dependencies with the corresponding feature flag:

```toml
cudarc = { version = "same as burn", features = ["cuda-11040"] }
```
