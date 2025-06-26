# Installation
Installing CubeCL is straightforward. It’s available as a Rust crate, and you can add it to your project by updating your Cargo.toml:

```toml
[dependencies]
cubecl = {
    version = "0.6.0",  # Replace with the latest version from crates.io
    features = ["wgpu"]  # Enable desired runtime features (e.g., wgpu, cuda, hip)
}
```

The more challenging aspect is ensuring that you have the necessary drivers to run the selected runtime.

CubeCL supports multiple GPU runtimes, each requiring specific drivers or frameworks. Enable the appropriate feature flag in Cargo.toml and ensure the corresponding drivers are installed.

| Platform | Runtime  | Supported OS                | Requirements                                             | Installation/Notes                                                                                                                                   | Feature Flag              |
|----------|----------|-----------------------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| WebGPU   | wgpu     | Linux, Windows, macOS, wasm | Vulkan drivers (typically pre-installed on modern OSes)  | On linux install the vulkan driver.                                                                                                                  | wgpu                      |
| CUDA     | CUDA     | Linux, Windows              | NVIDIA CUDA drivers and toolkit                          | Download and install from the NVIDIA CUDA Downloads page. Verify installation with nvidia-smi.                                                       | cuda                      |
| ROCm     | HIP      | Linux, Windows              | AMD ROCm framework                                       | Linux: Follow the ROCm Linux Quick Start. Windows: See the ROCm Windows Installation Guide.                                                          | hip                       |
| Metal    | wgpu     | macOS                       | Apple device with Metal support (macOS 10.13 or later)   | No additional drivers needed; Metal is built into macOS.                                                                                             | wgpu-msl                  |
| Vulkan   | wgpu     | Linux, Windows              | Vulkan drivers                                           | On linux install via package manager, on windows it is typically included with GPU drivers (NVIDIA/AMD).                                             | wgpu-spirv                |
