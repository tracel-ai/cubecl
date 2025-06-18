# Benchmarking reduction
Now that we have a basic understanding of how to perform a reduction operation, let's benchmark it to see how it performs in terms of speed and efficiency.

## Benchmarking struct
For benchmarking, we will create a struct that holds the necessary information for the benchmark, such as the input shape, device, and client. This struct will be used to run the benchmark tests and configure the benchmarking environment. Please note that the `Runtime` and `Float` traits are used to make the benchmark generic over different CubeCL runtimes and floating-point types. A `PhantomData` is used to indicate that the struct holds a type parameter `F` without actually storing a value of that type, which is useful for generic programming in Rust for more information see the [Rust documentation](https://doc.rust-lang.org/std/marker/struct.PhantomData.html).
```rust,ignore
# use std::marker::PhantomData;
#
# use cubecl::benchmark::{Benchmark, TimingMethod};
# use cubecl::server::Handle;
# use cubecl::std::tensor::compact_strides;
# use cubecl::{future, prelude::*};

pub struct ReductionBench<R: Runtime, F: Float> {
    input_shape: [usize; 2],
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _phantom_data: PhantomData<F>,
}
```
## Implementing the benchmark trait
To benchmark a CubeCL kernel, it is recommended to implement the `Benchmark` trait that defines the necessary methods for preparing, executing, and synchronizing the benchmark, because GPU are asynchronous and most benchmarking tool will not wait for the GPU to finish executing the kernel before measuring the time it took to execute it with a sync.
```rust,ignore
pub trait Benchmark {
    /// Benchmark arguments.
    type Args: Clone;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self) -> Self::Args;
    /// Execute the benchmark and returns the time it took to complete.
    fn execute(&self, args: Self::Args);
    /// Name of the benchmark, should be short and it should match the name
    /// defined in the crate Cargo.toml
    fn name(&self) -> String;
    /// Wait for computation to complete.
    fn sync(&self);
}
```

In the `prepare` method, we will create the input data and return a handle as a Args that will be used in the `execute` method. The `execute` method will launch the kernel and the sync method will wait for the GPU to finish executing the kernel before measuring the time it took to execute it. Please note that strides can be computed using the `compact_strides` function from the `cubecl::std::tensor` module, which will compute the strides for a given shape with a compact representation.
```rust,ignore
# use std::marker::PhantomData;
#
# use cubecl::benchmark::{Benchmark, TimingMethod};
# use cubecl::server::Handle;
# use cubecl::std::tensor::compact_strides;
# use cubecl::{future, prelude::*};
#
# pub struct ReductionBench<R: Runtime, F: Float> {
#     input_shape: [usize; 2],
#     device: R::Device,
#     client: ComputeClient<R::Server, R::Channel>,
#     _phantom_data: PhantomData<F>,
# }

impl<R: Runtime, F: Float> Benchmark for ReductionBench<R, F> {
    type Args = Handle;

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);

        let total_size: usize = self.input_shape.iter().product();
        let input: Vec<f32> = (0..total_size).into_iter().map(|i| i as f32).collect();

        client.create(f32::as_bytes(&input))
    }

    fn execute(&self, args: Self::Args) {
        let client = R::client(&self.device);

        let input_shape = &self.input_shape;
        let output_shape = &self.input_shape[0..1];

        let input_stride = compact_strides(input_shape);
        let output_stride = compact_strides(output_shape);

        let output_size: usize = output_shape.iter().product();
        let input = args;
        let output = client.empty(output_size * core::mem::size_of::<f32>());

        unsafe {
            reduce_matrix::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                TensorArg::from_raw_parts::<F>(&input, &input_stride, input_shape, 1),
                TensorArg::from_raw_parts::<F>(&output, &output_stride, output_shape, 1),
            );
        }
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!("{}-reduction-{:?}", R::name(&client), self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }
}
```

## Running the benchmark
Now that we have implemented the `Benchmark` trait, we can run the benchmark using the `Benchmark::run` method. This method will execute the benchmark and return the time it took to complete it.
```rust,ignore
# use std::marker::PhantomData;
#
# use cubecl::benchmark::{Benchmark, TimingMethod};
# use cubecl::server::Handle;
# use cubecl::std::tensor::compact_strides;
# use cubecl::{future, prelude::*};
#
# pub struct ReductionBench<R: Runtime, F: Float> {
#     input_shape: [usize; 2],
#     device: R::Device,
#     client: ComputeClient<R::Server, R::Channel>,
#     _phantom_data: PhantomData<F>,
# }
#
# impl<R: Runtime, F: Float> Benchmark for ReductionBench<R, F> {
#     type Args = Handle;
#
#     fn prepare(&self) -> Self::Args {
#         let client = R::client(&self.device);
#
#         let total_size: usize = self.input_shape.iter().product();
#         let input: Vec<f32> = (0..total_size).into_iter().map(|i| i as f32).collect();
#
#         client.create(f32::as_bytes(&input))
#     }
#
#     fn execute(&self, args: Self::Args) {
#         let client = R::client(&self.device);
#
#         let input_shape = &self.input_shape;
#         let output_shape = &self.input_shape[0..1];
#
#         let input_stride = compact_strides(input_shape);
#         let output_stride = compact_strides(output_shape);
#
#         let output_size: usize = output_shape.iter().product();
#         let input = args;
#         let output = client.empty(output_size * core::mem::size_of::<f32>());
#
#         unsafe {
#             reduce_matrix::launch_unchecked::<F, R>(
#                 &self.client,
#                 CubeCount::Static(1, 1, 1),
#                 CubeDim::new(1, 1, 1),
#                 TensorArg::from_raw_parts::<F>(&input, &input_stride, input_shape, 1),
#                 TensorArg::from_raw_parts::<F>(&output, &output_stride, output_shape, 1),
#             );
#         }
#     }
#
#     fn name(&self) -> String {
#         let client = R::client(&self.device);
#         format!("{}-reduction-{:?}", R::name(&client), self.input_shape).to_lowercase()
#     }
#
#     fn sync(&self) {
#         future::block_on(self.client.sync())
#     }
# }
#
# #[cube(launch_unchecked)]
# /// This function execute the reduction in the following way by reducing with a sum over each row
# /// [0 1 2]    [0 + 1 + 2]    [3 ]
# /// [3 4 5] -> [3 + 4 + 5] -> [12]
# /// [6 7 8]    [6 + 7 + 8]    [21]
# fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
#     for i in 0..input.shape(0) {
#         let mut acc = F::new(0.0f32);
#         for j in 0..input.shape(1) {
#             acc += input[i * input.stride(0) + j];
#         }
#         output[i] = acc;
#     }
# }
#
pub fn launch<R: Runtime, F: Float>(device: &R::Device) {
    let client = R::client(&device);

    let bench1 = ReductionBench::<R, F> {
        input_shape: [512, 8 * 1024],
        client: client.clone(),
        device: device.clone(),
        _phantom_data: PhantomData,
    };
    let bench2 = ReductionBench::<R, F> {
        input_shape: [128, 32 * 1024],
        client: client.clone(),
        device: device.clone(),
        _phantom_data: PhantomData,
    };

    for bench in [bench1, bench2] {
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::System));
    }
}

fn main() {
    launch::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
}
```

## The results
When you run the above code, it will execute the reduction benchmark for two different input shapes and print the results to the console. The output will look something like this:
```
wgpu<wgsl>-reduction-[512, 8192]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        240.730ms
  Variance    1.595µs
  Median      240.310ms
  Min         239.974ms
  Max         244.374ms
―――――――――――――――――――――――――
wgpu<wgsl>-reduction-[128, 32768]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        241.018ms
  Variance    1.068µs
  Median      240.943ms
  Min         239.734ms
  Max         243.782ms
―――――――――――――――――――――――――
```
Somehow our time is not that good, but it is expected because we are using a very simple kernel that does not take advantage of the GPU parallelism. In the next chapter, we will see how to optimize our kernel to take advantage of the GPU parallelism and improve the performance of our reduction operation.