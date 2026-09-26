// ANCHOR: implementation
use cubecl::{Device, prelude::*};
use cubecl_example::gpu_tensor::GpuTensor; // Change to the path of your own module containing the GpuTensor

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    for row in 0..input.shape(0) {
        let mut acc = F::new(0.0f32);
        for col in 0..input.shape(1) {
            acc += input[row * input.stride(0) + col];
        }
        output[row] = acc;
    }
}
// ANCHOR_END: implementation

// ANCHOR: launch
pub fn launch<F: Float + CubeElement>(device: &Device) {
    let client = device.client();

    let input = GpuTensor::<F>::arange(vec![3, 3], &client);
    let output = GpuTensor::<F>::empty(vec![3], &client);

    unsafe {
        reduce_matrix::launch_unchecked::<F>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            input.as_arg(),
            output.as_arg(),
        )
    };

    println!(
        "Executed reduction with runtime {:?} => {:?}",
        client.name(),
        output.read(&client)
    );
}

fn main() {
    launch::<f32>(&Default::default());
}
// ANCHOR_END: launch
