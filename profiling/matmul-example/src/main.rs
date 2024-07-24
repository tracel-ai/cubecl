fn main() {
    #[cfg(feature = "burn-tch-cuda")]
    tch_gpu::run();
    #[cfg(feature = "cube-cuda")]
    cube_cuda::run();
}

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
