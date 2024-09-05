# Cuda runtime

The runtime uses the lower level primitives from [cudarc](https://github.com/coreylowman/cudarc) to compile generated CUDA code into a ptx and execute it at runtime.

## Setup

By default, this runtime uses the latest CUDA version.
If the default features are not enabled, the CUDA version detected during build time will be used instead.
To specify a fixed CUDA version, disable the default features and add `cudarc` to your dependencies with the appropriate feature flag:

```toml
cudarc = { version = "same as burn", features = ["cuda-11040"] }
```
