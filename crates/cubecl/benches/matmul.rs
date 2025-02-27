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

        matmul::launch::<R, E>(&self.strategy, &self.client, lhs, rhs, out).unwrap();
    }

    fn name(&self) -> String {
        format!(
            "matmul-{}-{}-{:?}",
            R::name(),
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
        (1, 6144, 6144, 6144),
        (1, 5000, 5000, 5000),
        (2, 4096, 4096, 4096),
        (16, 6144, 2048, 513),
        (32, 256, 256, 256),
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

fn main() {
    #[cfg(feature = "wgpu")]
    {
        run::<cubecl::wgpu::WgpuRuntime, f32>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );
        run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), matmul::Strategy::PlaneMma);
    }

    #[cfg(feature = "wgpu-spirv")]
    {
        type R = cubecl::wgpu::WgpuRuntime;
        use half::f16;

        run::<R, f16>(Default::default(), matmul::Strategy::Simple);
        run::<R, f16>(Default::default(), matmul::Strategy::Specialized);
        run::<R, f16>(Default::default(), matmul::Strategy::DoubleBuffering);
        run::<R, flex32>(Default::default(), matmul::Strategy::Simple);
        run::<R, flex32>(Default::default(), matmul::Strategy::Specialized);
        run::<R, flex32>(Default::default(), matmul::Strategy::DoubleBuffering);
        run::<R, f32>(Default::default(), matmul::Strategy::Simple);
        run::<R, f32>(Default::default(), matmul::Strategy::Specialized);
        run::<R, f32>(Default::default(), matmul::Strategy::DoubleBuffering);
    }

    #[cfg(all(feature = "hip", target_os = "linux"))]
    {
        // TODO: unless annotated OOM, all the benches can randomly hang
        // Full-precision ----------------------------------------------------
        // Tiling2D
        run::<cubecl::hip::HipRuntime, f32>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );
        // PlaneMma
        // run::<cubecl::hip::HipRuntime, f32>(Default::default(), matmul::Strategy::PlaneMma);
        // CmmaOld
        // run::<cubecl::hip::HipRuntime,<cubecl::hip::HipDialect> f32>(Default::default(), matmul::Strategy::CmmaOld(Default::default()));
        // Accelerated
        run::<cubecl::hip::HipRuntime, f32>(Default::default(), matmul::Strategy::Simple);
        // Half-precision ----------------------------------------------------
        // Tiling2D
        run::<cubecl::hip::HipRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::Tiling2D(Default::default()),
        );
        // PlaneMma: OOM
        // run::<cubecl::hip::HipRuntime, half::f16>(Default::default(), matmul::Strategy::PlaneMma);
        // CmmaOld: OOM
        // run::<cubecl::hip::HipRuntime, half::f16>(Default::default(), matmul::Strategy::CmmaOld(Default::default()));
        // Accelerated
        run::<cubecl::hip::HipRuntime, half::f16>(Default::default(), matmul::Strategy::Simple);
    }

    #[cfg(feature = "cuda")]
    {
        use half::f16;

        run::<cubecl::cuda::CudaRuntime, f16>(Default::default(), matmul::Strategy::Simple);
        // run::<cubecl::cuda::CudaRuntime, f16>(
        //     Default::default(),
        //     matmul::Strategy::SimpleBarrier(matmul::SimpleBarrierLoadingStrategy::Elected),
        // );
        // run::<cubecl::cuda::CudaRuntime, f16>(
        //     Default::default(),
        //     matmul::Strategy::SimpleBarrier(matmul::SimpleBarrierLoadingStrategy::ElectedOnly),
        // );
        // run::<cubecl::cuda::CudaRuntime, f16>(
        //     Default::default(),
        //     matmul::Strategy::SimpleBarrier(matmul::SimpleBarrierLoadingStrategy::SplitUnit),
        // );
        // run::<cubecl::cuda::CudaRuntime, f16>(
        //     Default::default(),
        //     matmul::Strategy::SimpleBarrier(matmul::SimpleBarrierLoadingStrategy::Cyclic),
        // );
        // run::<cubecl::cuda::CudaRuntime, f16>(
        //     Default::default(),
        //     matmul::Strategy::SimpleBarrier(matmul::SimpleBarrierLoadingStrategy::SplitPlane),
        // );
        // run::<cubecl::cuda::CudaRuntime, f16>(
        //     Default::default(),
        //     matmul::Strategy::SimplePipelined,
        // );
        // run::<cubecl::cuda::CudaRuntime, f16>(Default::default(), matmul::Strategy::Specialized);
        // run::<cubecl::cuda::CudaRuntime, f16>(
        //     Default::default(),
        //     matmul::Strategy::DoubleBuffering,
        // );

        // run::<cubecl::cuda::CudaRuntime, flex32>(Default::default(), matmul::Strategy::Simple);
        // run::<cubecl::cuda::CudaRuntime, flex32>(
        //     Default::default(),
        //     matmul::Strategy::SimpleBarrier,
        // );
        // run::<cubecl::cuda::CudaRuntime, flex32>(
        //     Default::default(),
        //     matmul::Strategy::SimplePipelined,
        // );
        // run::<cubecl::cuda::CudaRuntime, flex32>(Default::default(), matmul::Strategy::Specialized);
        // run::<cubecl::cuda::CudaRuntime, flex32>(
        //     Default::default(),
        //     matmul::Strategy::DoubleBuffering,
        // );

        // run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::Simple);
        // run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::SimpleBarrier);
        // run::<cubecl::cuda::CudaRuntime, f32>(
        //     Default::default(),
        //     matmul::Strategy::SimplePipelined,
        // );
        // run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::Specialized);
        // run::<cubecl::cuda::CudaRuntime, f32>(
        //     Default::default(),
        //     matmul::Strategy::DoubleBuffering,
        // );
    }
}
