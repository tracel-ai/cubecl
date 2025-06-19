# Vectorized Reduction
In this section, we will explore how to implement a vectorized reduction operation using CubeCL. Vectorization is a powerful technique that allows us to process multiple data elements simultaneously, significantly improving performance for certain types of computations.

## What is vectorization?
Vectorization is the process of converting scalar operations (which operate on single data elements) into vector operations (which operate on multiple data elements simultaneously). This is typically done using SIMD (Single Instruction, Multiple Data) instructions available in modern CPUs and GPUs. By leveraging vectorization, we can achieve significant performance improvements for operations that can be vectorized. For more information on vectorization in CubeCL, you can refer to [this section](../core-features/vectorization.md).

## Application to the reduction problem
To apply vectorization to the reduction problem, we will modify our reduction kernel to process multiple elements at once. This means that instead of summing one element at a time, we will sum multiple elements with vectorization, which can lead to substantial performance gains.


```rust,ignore
# use std::marker::PhantomData;
#
# use cubecl::benchmark::{Benchmark, TimingMethod};
# use cubecl::server::Handle;
# use cubecl::std::tensor::compact_strides;
# use cubecl::{future, prelude::*};
#
# pub struct ReductionBench<R: Runtime, F: Float> {
#    input_shape: [usize; 2],
#    device: R::Device,
#    client: ComputeClient<R::Server, R::Channel>,
#    _phantom_data: PhantomData<F>,
# }
#
// ...
const VECTORIZATION_FACTOR: u32 = 4;

impl<R: Runtime, F: Float> Benchmark for ReductionBench<R, F> {
#     type Input = Handle;
#     type Output = Handle;
#
#     fn prepare(&self) -> Self::Input {
#         let client = R::client(&self.device);
#
#         let total_size: usize = self.input_shape.iter().product();
#         let input: Vec<f32> = (0..total_size).into_iter().map(|i| i as f32).collect();
#
#         client.create(f32::as_bytes(&input))
#     }
    // ...
    fn execute(&self, args: Self::Input) -> Self::Output {
#         let client = R::client(&self.device);
#
#         let input_shape = &self.input_shape;
#         let output_shape = &self.input_shape[0..1];
#
#         let input_stride = &compact_strides(input_shape);
#         let output_stride = &compact_strides(output_shape);
#
#         let output_size: usize = output_shape.iter().product();
#         let input = args;
#         let output = client.empty(output_size * core::mem::size_of::<f32>());
        // ...
        unsafe {
            reduce_matrix::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(input_shape[0] as u32, 1, 1),
                TensorArg::from_raw_parts::<F>(
                    &input,
                    input_stride,
                    input_shape,
                    VECTORIZATION_FACTOR as u8, // Add the vectorization factor here
                ),
                TensorArg::from_raw_parts::<F>(
                    &output,
                    output_stride,
                    output_shape,
                    VECTORIZATION_FACTOR as u8, // Add the vectorization factor here
                ),
            );
        }

        output
    }
    // ...
#     fn name(&self) -> String {
#         let client = R::client(&self.device);
#         format!("{}-reduction-{:?}", R::name(&client), self.input_shape).to_lowercase()
#     }
#
#     fn sync(&self) {
#         future::block_on(self.client.sync())
#     }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut acc = Line::new(F::new(0.0f32));
    for i in 0..input.shape(1) / VECTORIZATION_FACTOR { // Because we iterate four at a time, we needs to divide the upper_bound by four.
        acc = acc + input[UNIT_POS_X * input.stride(0) + i];
    }
    output[UNIT_POS_X] = acc;
}

# pub fn launch<R: Runtime, F: Float>(device: &R::Device) {
#     let client = R::client(&device);
#
#     let bench1 = ReductionBench::<R, F> {
#         input_shape: [512, 8 * 1024],
#         client: client.clone(),
#         device: device.clone(),
#         _phantom_data: PhantomData,
#     };
#     let bench2 = ReductionBench::<R, F> {
#         input_shape: [128, 32 * 1024],
#         client: client.clone(),
#         device: device.clone(),
#         _phantom_data: PhantomData,
#     };
#
#     for bench in [bench1, bench2] {
#         println!("{}", bench.name());
#         println!("{}", bench.run(TimingMethod::System));
#     }
# }
#
# fn main() {
#     launch::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
# }
```

