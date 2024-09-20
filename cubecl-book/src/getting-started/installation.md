# Installation

Installing CubeCL is straightforward. It’s available as a Rust crate, and you can add it to your
project by updating your `Cargo.toml`:

```toml
[dependencies]
cubecl = { version = "{version}", features = ["cuda", "wgpu"] }
```

The more challenging aspect is ensuring that you have the necessary drivers to run the selected
runtime.

For `wgpu` on Linux and Windows, Vulkan drivers are required. These drivers are usually included
with the default OS installation. However, on certain setups, such as Windows Subsystem for Linux
(WSL), you may need to install them manually if they are missing.

For `cuda`, simply install the CUDA drivers on your device. You can follow the installation
instructions provided on the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
