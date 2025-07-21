use cubecl::prelude::*;
use cubecl_example::gpu_tensor::GpuTensor; // Change to the path of your own module containing the GpuTensor

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    for i in 0..input.shape(0) {
        let mut acc = F::new(0.0f32);
        for j in 0..input.shape(1) {
            acc += input[i * input.stride(0) + j];
        }
        output[i] = acc;
    }
}

pub fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);

    let input = GpuTensor::<R, F>::arange(vec![3, 3], &client);
    let output = GpuTensor::<R, F>::empty(vec![3, 3], &client);

    unsafe {
        reduce_matrix::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            input.into_tensor_arg(1),
            output.into_tensor_arg(1),
        )
    };

    println!(
        "Executed reduction with runtime {:?} => {:?}",
        R::name(&client),
        output.read(&client)
    );
}

fn main() {
    launch::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
}
