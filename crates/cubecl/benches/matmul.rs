use cubecl::prelude::*;
use cubecl_linalg::matmul;
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

        matmul::launch::<R, E>(&self.strategy, &self.client, lhs, rhs, out);
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn name(&self) -> String {
        format!("matmul-{}-{}-{:?}", R::name(), E::as_elem(), self.strategy).to_lowercase()
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
    client.enable_timestamps();

    let bench = MatmulBench::<R, E> {
        b: 1,
        m: 2048,
        k: 2048,
        n: 2048,
        client,
        device,
        strategy,
        _e: PhantomData,
    };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::DeviceOnly));
}

fn main() {
    #[cfg(feature = "wgpu")]
    {
        run::<cubecl::wgpu::WgpuRuntime, f32>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );
        run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), matmul::Strategy::PlaneMma);
    }

    #[cfg(all(feature = "hip", target_os = "linux"))]
    {
        // TODO: randomly hangs on sync after kernel launch
        run::<cubecl::hip::HipRuntime, f32>(
            Default::default(),
            MatmulKind::Cmma,
        );
        run::<cubecl::hip::HipRuntime, f32>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );

        run::<cubecl::hip::HipRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );
        // TODO: OOM
        run::<cubecl::hip::HipRuntime, half::f16>(
            Default::default(),
            MatmulKind::Cmma,
        );
    }

    #[cfg(feature = "cuda")]
    {
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );
        run::<cubecl::cuda::CudaRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            matmul::Strategy::CmmaOld(Default::default()),
        );
        run::<cubecl::cuda::CudaRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::CmmaOld(Default::default()),
        );
        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::PlaneMma);
        run::<cubecl::cuda::CudaRuntime, half::f16>(Default::default(), matmul::Strategy::PlaneMma);
        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::Accelerated);
        run::<cubecl::cuda::CudaRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::Accelerated,
        );
    }
}
