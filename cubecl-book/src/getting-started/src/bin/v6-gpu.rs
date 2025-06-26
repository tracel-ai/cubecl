use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::{future, prelude::*};
use cubecl_example::gpu_tensor::GpuTensor; // Change to the path of your own module containing the GpuTensor

pub struct ReductionBench<R: Runtime, F: Float + CubeElement> {
    input_shape: Vec<usize>,
    client: ComputeClient<R::Server, R::Channel>,
    _f: PhantomData<F>,
}

const LINE_SIZE: u32 = 4;

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
                CubeDim::new(self.input_shape[0] as u32, self.input_shape[1] as u32, 1),
                input.into_tensor_arg(LINE_SIZE as u8),
                output.into_tensor_arg(LINE_SIZE as u8),
            );
        }

        output
    }
}

// Note the addition of the [Line] struct inside the tensor to guarantee that the data is contiguous and can be parallelized.
#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut acc = Line::new(F::new(0.0f32)); // A [Line] is also necessary here
    for i in 0..input.shape(2) / LINE_SIZE {
        acc = acc + input[UNIT_POS_X * input.stride(0) + UNIT_POS_Y * input.stride(1) + i];
    }
    output[UNIT_POS_X * input.stride(0) + UNIT_POS_Y] = acc;
}

pub fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(&device);

    let bench1 = ReductionBench::<R, F> {
        input_shape: vec![64, 256, 1024],
        client: client.clone(),
        _f: PhantomData,
    };
    let bench2 = ReductionBench::<R, F> {
        input_shape: vec![64, 64, 4096],
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
