use cubecl::prelude::*;
use std::marker::PhantomData;

use cubecl::benchmark::Benchmark;
use cubecl::client::SyncType;
use cubecl::frontend::Float;
use cubecl_linalg::matmul::cmma::matmul_cmma;
use cubecl_linalg::matmul::tiling2d::matmul_tiling_2d;
use cubecl_linalg::tensor::TensorHandle;

impl<R: Runtime, E: Float> Benchmark for Tiling2dBench<R, E> {
    type Args = (TensorHandle<R, E>, TensorHandle<R, E>, TensorHandle<R, E>);

    fn prepare(&self) -> Self::Args {
        let (b, m, k, n) = (self.b, self.m, self.k, self.n);
        let client = R::client(&self.device);
        let lhs = TensorHandle::zeros(client.clone(), vec![b, m, k]);
        let rhs = TensorHandle::zeros(client.clone(), vec![b, k, n]);
        let out = TensorHandle::zeros(client.clone(), vec![b, m, n]);

        (lhs, rhs, out)
    }

    fn execute(&self, (lhs, rhs, out): Self::Args) {
        match self.kind {
            MatmulKind::Tiling2d => {
                matmul_tiling_2d(lhs, rhs, out, Default::default(), &self.device);
            }
            MatmulKind::Cmma => {
                matmul_cmma(lhs, rhs, out, &self.device);
            }
        }
    }

    fn name(&self) -> String {
        let elem = E::as_elem();
        format!("tiling2d-{}-{:?}-{:?}", R::name(), elem, self.kind)
    }

    fn sync(&self) {
        let client = R::client(&self.device);
        client.sync(SyncType::Wait);
    }
}

#[allow(dead_code)]
struct Tiling2dBench<R: Runtime, E> {
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    kind: MatmulKind,
    device: R::Device,
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
    let bench = Tiling2dBench::<R, E> {
        b: 32,
        m: 1024,
        k: 1024,
        n: 1024,
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
    run::<cubecl::cuda::CudaRuntime, F32>(Default::default(), MatmulKind::Cmma);
}
