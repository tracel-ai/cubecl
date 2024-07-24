# CubeCL Profiling Guide

## 1. Ensure the CUDA Runtime is Installed

To profile your GPU kernels, you must have the CUDA runtime installed on your system. Follow the official NVIDIA documentation to install the CUDA toolkit and runtime for your operating system: [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## 2. Ensure NVIDIA Nsight Compute is Installed

NVIDIA Nsight Compute is a powerful tool for GPU profiling. Make sure it is installed on your system. You can download and install it from the NVIDIA Developer website: [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute).

## 3. Isolate the Kernel to be Profiled into a Main Function

For effective profiling, isolate the kernel you want to profile into a main function. This allows you to focus on the performance of a specific kernel without interference from other parts of your code.

## 4. Use the CUDA device/runtime
Make sure your code uses the CUDA runtime API and device for lauching the kernel.

```rust
#[cfg(feature = "cube-cuda")]
mod cube_cuda {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    use cubecl::frontend::F32;
    use cubecl::linalg::{matmul::tiling2d, tensor::TensorHandle};
    use cubecl::prelude::*;
    use cubecl::Runtime;

    pub fn run() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        let num_of_batch = 12;
        let heigth = 1024;
        let width = 1024;

        let tensor_values: Vec<f32> = (0..num_of_batch * heigth * width)
            .map(|x| x as f32)
            .collect();
        let tensor_a_handle = client.create(f32::as_bytes(&tensor_values));
        let tensor_b_handle = client.create(f32::as_bytes(&tensor_values));
        let tensor_c_handle = client.empty(12 * 1024 * 1024 * core::mem::size_of::<f32>());

        let tensor_a_shape = vec![num_of_batch, heigth, width];
        let tensor_b_shape = vec![num_of_batch, heigth, width];
        let tensor_c_shape = vec![num_of_batch, heigth, width];

        let tensor_a: TensorHandle<CudaRuntime, F32> =
            TensorHandle::new_contiguous(tensor_a_shape, tensor_a_handle);
        let tensor_b: TensorHandle<CudaRuntime, F32> =
            TensorHandle::new_contiguous(tensor_b_shape, tensor_b_handle);
        let tensor_c: TensorHandle<CudaRuntime, F32> =
            TensorHandle::new_contiguous(tensor_c_shape, tensor_c_handle);
        tiling2d::launch(&client, tensor_a, tensor_b, tensor_c, Default::default());
    }
}
```

## 5. Building an executable
Compile your main function into an executable using cargo, in the last examples case it would be by running the command : 
cargo build --release --features cube-cuda

## 6. Ensure NVIDIA Nsight Compute has permission for Performance counter
This can be done by using sudo or by modifying Kernel Module Parameters as described at [link](https://gist.github.com/xaliander/8173ffe623546529c99e9cdd7e0655c4)

## 7. Follow the NVIDIA Nsight Compute guide
To profile and interpret the profilling results refer to the following [nvidia guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

## Optional : Profiling Burn.
To profile Burn operations, use a runtime which uses Cuda, such as LibTorch with a Cuda device or Cuda-Jit

Example:
```rust
#[cfg(feature = "burn-tch-cuda")]
mod tch_gpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice, TchTensor};
    use burn::tensor::{Distribution, Tensor};

    pub fn run() {
        let device = LibTorchDevice::Cuda(0);
        let tensor_1: Tensor<LibTorch, 3> =
            Tensor::<LibTorch, 3>::random([12, 1024, 1024], Distribution::Default, &device);
        let tensor_2: Tensor<LibTorch, 3> =
            Tensor::<LibTorch, 3>::random([12, 1024, 1024], Distribution::Default, &device);
        let output = tensor_1.matmul(tensor_2);
    }
}
```
