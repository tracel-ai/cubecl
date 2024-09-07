use cubecl::prelude::*;
use std::marker::PhantomData;

use cubecl::benchmark::Benchmark;
use cubecl::client::SyncType;
use cubecl::frontend::Float;
use cubecl_linalg::matmul;
use cubecl_linalg::tensor::TensorHandle;

impl<R: Runtime, E: Float> Benchmark for MatmulBench<R, E> {
    type Args = (TensorHandle<R, E>, TensorHandle<R, E>, TensorHandle<R, E>);

    fn prepare(&self) -> Self::Args {
        let (b, m, k, n) = (self.b, self.m, self.k, self.n);
        let client = R::client(&self.device);
        let lhs = TensorHandle::zeros(&client, vec![b, m, k]);
        let rhs = TensorHandle::zeros(&client, vec![b, k, n]);
        let out = TensorHandle::zeros(&client, vec![b, m, n]);

        (lhs, rhs, out)
    }

    fn execute(&self, (lhs, rhs, out): Self::Args) {
        match self.kind {
            MatmulKind::Tiling2d => {
                matmul::tiling2d::launch(&self.client, lhs, rhs, out, Default::default());
            }
            MatmulKind::Cmma => {
                matmul::cmma::launch(&self.client, lhs, rhs, out);
            }
        }
    }

    fn num_samples(&self) -> usize {
        100
    }

    fn name(&self) -> String {
        format!("matmul-{}-{}-{:?}", R::name(), E::as_elem(), self.kind).to_lowercase()
    }

    fn sync(&self) {
        self.client.sync(SyncType::Wait);
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
    let bench = MatmulBench::<R, E> {
        b: 32,
        m: 1024,
        k: 1024,
        n: 1024,
        client: R::client(&device),
        device,
        kind,
        _e: PhantomData,
    };
    println!("{}", bench.name());
    println!("{}", bench.run());
}

fn main() {
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, F32>(Default::default(), MatmulKind::Tiling2d);

    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, F32>(Default::default(), MatmulKind::Tiling2d);
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, F16>(Default::default(), MatmulKind::Tiling2d);
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, F32>(Default::default(), MatmulKind::Cmma);
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, F16>(Default::default(), MatmulKind::Cmma);
}
