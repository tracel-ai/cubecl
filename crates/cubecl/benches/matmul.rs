use cubecl::prelude::*;
use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimestampsResult, TimingMethod};
use cubecl::frontend::Float;
use cubecl::future;
use cubecl_linalg::matmul;
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

        match self.kind {
            MatmulKind::Tiling2d => {
                matmul::tiling2d::launch(&self.client, lhs, rhs, out, Default::default());
            }
            MatmulKind::Cmma => {
                matmul::cmma::launch(&self.client, lhs, rhs, out, Default::default());
            }
        }
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn name(&self) -> String {
        format!("matmul-{}-{}-{:?}", R::name(), E::as_elem(), self.kind).to_lowercase()
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
    kind: MatmulKind,
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
fn run<R: Runtime, E: Float>(device: R::Device, kind: MatmulKind) {
    let client = R::client(&device);
    client.enable_timestamps();

    let bench = MatmulBench::<R, E> {
        b: 1,
        m: 2048,
        k: 2048,
        n: 2048,
        client,
        device,
        kind,
        _e: PhantomData,
    };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::DeviceOnly));
}

fn main() {
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), MatmulKind::Tiling2d);

    #[cfg(feature = "hip")]
    {
        run::<cubecl::hip::HipRuntime, f32>(Default::default(), MatmulKind::Tiling2d);
        run::<cubecl::hip::HipRuntime, half::f16>(Default::default(), MatmulKind::Tiling2d);
    }

    #[cfg(feature = "cuda")]
    {
        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), MatmulKind::Tiling2d);
        run::<cubecl::cuda::CudaRuntime, half::f16>(Default::default(), MatmulKind::Tiling2d);
        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), MatmulKind::Cmma);
        run::<cubecl::cuda::CudaRuntime, half::f16>(Default::default(), MatmulKind::Cmma);
    }
}
