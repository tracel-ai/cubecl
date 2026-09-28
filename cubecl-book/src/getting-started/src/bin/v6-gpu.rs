// ANCHOR: implementation
// ANCHOR: benchmark_struct
use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::{Device, future, prelude::*};
use cubecl_example::gpu_tensor::GpuTensor; // Change to the path of your own module containing the GpuTensor

pub struct ReductionBench<F: Float + CubeElement> {
    input_shape: Vec<usize>,
    client: Client,
    _f: PhantomData<F>,
}

const VECTOR_SIZE: usize = 4;
// ANCHOR_END: benchmark_struct

impl<F: Float + CubeElement> Benchmark for ReductionBench<F> {
    type Input = GpuTensor<F>;
    type Output = GpuTensor<F>;

    fn prepare(&self) -> Self::Input {
        GpuTensor::<F>::arange(self.input_shape.clone(), &self.client)
    }

    fn name(&self) -> String {
        format!("{}-reduction-{:?}", self.client.name(), self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("Failed to synchronize device")
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let output_shape: Vec<usize> = self.input_shape[..2].to_vec();
        let output = GpuTensor::<F>::empty(output_shape, &self.client);

        unsafe {
            reduce_matrix::launch_unchecked::<F>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(self.input_shape[0] as u32, self.input_shape[1] as u32),
                VECTOR_SIZE,
                input.as_arg(),
                output.as_arg(),
            );
        }

        Ok(output)
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float, N: Size>(input: &Tensor<Vector<F, N>>, output: &mut Tensor<F>) {
    let x = UNIT_POS_X as usize;
    let y = UNIT_POS_Y as usize;
    let offset = (x * input.stride(0) + y * input.stride(1)) / N::value();
    let mut acc = Vector::<F, N>::new(F::new(0.0f32));
    for i in 0..input.shape(2) / N::value() {
        acc += input[offset + i];
    }
    let row = x * output.stride(0) + y;
    output[row] = acc.vector_sum();
}
// ANCHOR_END: implementation

// ANCHOR: launch
pub fn launch<F: Float + CubeElement>(device: &Device) {
    let client = device.client();

    let bench1 = ReductionBench::<F> {
        input_shape: vec![64, 256, 1024],
        client: client.clone(),
        _f: PhantomData,
    };
    let bench2 = ReductionBench::<F> {
        input_shape: vec![64, 64, 4096],
        client: client.clone(),
        _f: PhantomData,
    };

    for bench in [bench1, bench2] {
        println!("{}", bench.name());
        println!(
            "{}",
            bench.run(TimingMethod::System).expect("Benchmark failed")
        );
    }
}

fn main() {
    launch::<f32>(&Default::default());
}
// ANCHOR_END: launch
