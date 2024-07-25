// fn main() {
//     #[cfg(feature = "cuda")]
//     gelu::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
//     #[cfg(feature = "wgpu")]
//     gelu::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
// }

use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::marker::PhantomData;

use cubecl::benchmark::Benchmark;
use cubecl::client::SyncType;
use cubecl::frontend::Float;
use cubecl::linalg::tensor::TensorHandle;

#[cube(launch)]
fn execute<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>, out: &mut Tensor<F>) {
    if ABSOLUTE_POS < out.len() {
        for i in range(0, 256, Comptime::new(false)) {
            if i % UInt::new(2) == UInt::new(0) {
                out[ABSOLUTE_POS] -= F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            } else {
                out[ABSOLUTE_POS] += F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            }
        }
    }
}

impl<R: Runtime, E: Float> Benchmark for UnaryBench<R, E> {
    type Args = (TensorHandle<R, E>, TensorHandle<R, E>, TensorHandle<R, E>);

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);
        let lhs = TensorHandle::zeros(&client, self.shape.clone());
        let rhs = TensorHandle::zeros(&client, self.shape.clone());
        let out = TensorHandle::zeros(&client, self.shape.clone());

        (lhs, rhs, out)
    }

    fn execute(&self, (lhs, rhs, out): Self::Args) {
        let num_elems: usize = out.shape.iter().product();

        let cube_count =
            calculate_cube_count_elemwise::<R::Server>(num_elems / self.vectorization as usize, 16);

        execute::launch::<E, R>(
            &self.client,
            cube_count,
            CubeDim::new(16, 16, 1),
            TensorArg::vectorized(self.vectorization, &lhs.handle, &lhs.strides, &lhs.shape),
            TensorArg::vectorized(self.vectorization, &rhs.handle, &rhs.strides, &rhs.shape),
            TensorArg::vectorized(self.vectorization, &out.handle, &out.strides, &out.shape),
        )
    }

    fn num_samples(&self) -> usize {
        100
    }

    fn name(&self) -> String {
        format!(
            "unary-{}-{}-{:?}",
            R::name(),
            E::as_elem(),
            self.vectorization
        )
        .to_lowercase()
    }

    fn sync(&self) {
        self.client.sync(SyncType::Wait);
    }
}

#[allow(dead_code)]
struct UnaryBench<R: Runtime, E> {
    shape: Vec<usize>,
    vectorization: u8,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum MatmulKind {
    Tiling2d,
    Cmma,
}

#[allow(dead_code)]
fn run<R: Runtime, E: Float>(device: R::Device, vectorization: u8) {
    let bench = UnaryBench::<R, E> {
        shape: vec![32, 512, 2048],
        vectorization,
        client: R::client(&device),
        device,
        _e: PhantomData,
    };
    println!("{}", bench.name());
    println!("{}", bench.run());
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, F16>(Default::default(), 8);
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, F32>(Default::default(), 4);
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, F32>(Default::default(), 1);
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, F32>(Default::default(), 4);
}
