# Installation
Installing CubeCL is straightforward. It’s available as a Rust crate, and you can add it to your project by updating your Cargo.toml:

```toml
[dependencies]
cubecl = {
    version = "0.5.0",  # Replace with the latest version from crates.io
    features = ["wgpu"]  # Enable desired runtime features (e.g., wgpu, cuda, hip)
}
```

The more challenging aspect is ensuring that you have the necessary drivers to run the selected runtime.

CubeCL supports multiple GPU runtimes, each requiring specific drivers or frameworks. Enable the appropriate feature flag in Cargo.toml and ensure the corresponding drivers are installed.

| Platform | Runtime  | Supported OS              | Requirements                                             | Installation/Notes                                                                                                                                   | Feature Flag              |
|----------|----------|---------------------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| WebGPU   | wgpu     | Linux, Windows, macOS     | Vulkan drivers (typically pre-installed on modern OSes)  | On WSL, manually install Vulkan drivers if missing. Refer to Microsoft’s WSL documentation. macOS uses Metal via WGPU; no additional drivers needed. | wgpu                      |
| CUDA     | CUDA     | Linux, Windows            | NVIDIA CUDA drivers and toolkit                          | Download and install from the NVIDIA CUDA Downloads page. Verify installation with nvidia-smi.                                                       | cuda                      |
| ROCm     | HIP      | Linux, Windows            | Vulkan drivers and AMD ROCm framework                    | Linux: Follow the ROCm Linux Quick Start. Windows: See the ROCm Windows Installation Guide. Ensure Vulkan drivers are installed.                     | hip                       |
| Metal    | wgpu     | macOS                     | Apple device with Metal support (macOS 10.13 or later)   | No additional drivers needed; Metal is built into macOS.                                                                                             | wgpu-msl                  |
| Vulkan   | wgpu     | Linux, Windows            | Vulkan drivers                                           | Linux: Install via package manager (e.g., apt install mesa-vulkan-drivers on Ubuntu). Windows: Typically included with GPU drivers (NVIDIA/AMD).     | wgpu-spirv                |
