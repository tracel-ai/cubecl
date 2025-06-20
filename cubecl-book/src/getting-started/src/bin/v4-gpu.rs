use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::{future, prelude::*};
use cubecl_example::gpu_tensor::GpuTensor; // Change to the path of your own module containing the GpuTensor

pub struct ReductionBench<R: Runtime, F: Float + CubeElement> {
    input_shape: Vec<usize>,
    client: ComputeClient<R::Server, R::Channel>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: Float + CubeElement> Benchmark for ReductionBench<R, F> {
    type Input = GpuTensor<R, F>;
    type Output = GpuTensor<R, F>;

    fn prepare(&self) -> Self::Input {
        GpuTensor::<R, F>::arange(self.input_shape.clone(), &self.client)
    }

    fn name(&self) -> String {
        format!("{}-reduction-{:?}", R::name(&self.client), self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        let output_shape: Vec<usize> = vec![self.input_shape[0]];
        let output = GpuTensor::<R, F>::empty(output_shape, &self.client);

        unsafe {
            reduce_matrix::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(self.input_shape[0] as u32, 1, 1), // Add parallelization on the first dimension
                input.into_tensor_arg(1),
                output.into_tensor_arg(1),
            );
        }

        output
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let mut acc = F::new(0.0f32);
    for i in 0..input.shape(1) {
        acc += input[UNIT_POS_X * input.stride(0) + i];
    }
    output[UNIT_POS_X] = acc;
}

pub fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(&device);

    let bench1 = ReductionBench::<R, F> {
        input_shape: vec![512, 8 * 1024],
        client: client.clone(),
        _f: PhantomData,
    };
    let bench2 = ReductionBench::<R, F> {
        input_shape: vec![128, 32 * 1024],
        client: client.clone(),
        _f: PhantomData,
    };

    for bench in [bench1, bench2] {
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::System));
    }
}

fn main() {
    launch::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
}
