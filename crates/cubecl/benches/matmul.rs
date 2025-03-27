use cubecl::prelude::*;
use cubecl_linalg::matmul::{self};
use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimestampsResult, TimingMethod};
use cubecl::frontend::Float;
use cubecl::future;
use cubecl_linalg::tensor::TensorHandle;

impl<R: Runtime, E: Float> Benchmark for MatmulBench<R, E> {
    type Args = (TensorHandle<R, E>, TensorHandle<R, E>);

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);

        let lhs = TensorHandle::zeros(&client, vec![self.b, self.m, self.k]);
        let rhs = TensorHandle::zeros(&client, vec![self.b, self.k, self.n]);

        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        let client = R::client(&self.device);
        let out = TensorHandle::empty(&client, vec![self.b, self.m, self.n]);

        matmul::launch::<R, E>(&self.strategy, &self.client, lhs, rhs, out).unwrap();
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);

        format!(
            "matmul-{}-{}-{:?}",
            R::name(&client),
            E::as_elem_native_unchecked(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn sync_elapsed(&self) -> TimestampsResult {
        future::block_on(self.client.sync_elapsed())
    }
}

#[allow(dead_code)]
struct MatmulBench<R: Runtime, E> {
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    strategy: matmul::Strategy,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
fn run<R: Runtime, E: Float>(device: R::Device, strategy: matmul::Strategy) {
    let client = R::client(&device);

    for (b, m, n, k) in [
        // (1, 6144, 6144, 16384),
        (1, 6144, 6144, 6144),
        (1, 5000, 5000, 5000),
        (2, 4096, 4096, 4096),
        // (16, 6144, 2048, 513),
        // (32, 256, 256, 256),
    ] {
        let bench = MatmulBench::<R, E> {
            b,
            m,
            k,
            n,
            client: client.clone(),
            device: device.clone(),
            strategy: strategy.clone(),
            _e: PhantomData,
        };
        println!("b: {b} m: {m} n: {n} k: {k}");
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::Full));
    }
}

#[allow(unused)]
fn run_benches<R: Runtime, E: Float>() {
    run::<R, E>(Default::default(), matmul::Strategy::DoubleBuffering);
    // run::<R, E>(
    //     Default::default(),
    //     matmul::Strategy::Simple(SyncLoadingStrategy::Cyclic),
    // );
    // run::<R, E>(
    //     Default::default(),
    //     matmul::Strategy::Tiling2D(Default::default()),
    // );
    // run::<cubecl::cuda::CudaRuntime, f16>(
    //     Default::default(),
    //     matmul::Strategy::Simple(SyncLoadingStrategy::Strided),
    // );
    // run::<cubecl::cuda::CudaRuntime, f16>(
    //     Default::default(),
    //     matmul::Strategy::SimpleBarrier(AsyncLoadingStrategy::Cooperative),
    // );
    // run::<cubecl::cuda::CudaRuntime, f16>(
    //     Default::default(),
    //     matmul::Strategy::SimpleBarrier(AsyncLoadingStrategy::Cyclic),
    // );
    // run::<cubecl::cuda::CudaRuntime, f16>(
    //     Default::default(),
    //     matmul::Strategy::SimpleBarrier(AsyncLoadingStrategy::Tma),
    // );
}

fn main() {
    #[cfg(feature = "wgpu")]
    {
        run_benches::<cubecl::wgpu::WgpuRuntime, f32>();
    }

    #[cfg(feature = "wgpu-spirv")]
    {
        run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
    }

    #[cfg(all(feature = "hip", target_os = "linux"))]
    {
        run_benches::<cubecl::hip::HipRuntime, half::f16>();
    }

    #[cfg(feature = "cuda")]
    {
        run_benches::<cubecl::cuda::CudaRuntime, half::f16>();
    }
}
