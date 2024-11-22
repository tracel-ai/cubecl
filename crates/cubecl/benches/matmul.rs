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
        b: 2,
        m: 4096,
        k: 4096,
        n: 4096,
        client,
        device,
        strategy,
        _e: PhantomData,
    };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::DeviceOnly));
}

fn main() {
    // #[cfg(feature = "wgpu")]
    // {
    //     run::<cubecl::wgpu::WgpuRuntime, f32>(
    //         Default::default(),
    //         matmul::Strategy::Tiling2D(Default::default()),
    //     );
    //     run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), matmul::Strategy::PlaneMma);
    // }

    #[cfg(feature = "wgpu-spirv")]
    {
        type R = cubecl::wgpu::WgpuRuntime<cubecl::wgpu::spirv::SpirvCompiler>;
        // run::<cubecl::wgpu::WgpuRuntime<cubecl::wgpu::spirv::SpirvCompiler>, f32>(
        //     Default::default(),
        //     matmul::Strategy::Tiling2D(Default::default()),
        // );
        // run::<cubecl::wgpu::WgpuRuntime<cubecl::wgpu::spirv::SpirvCompiler>, f32>(
        //     Default::default(),
        //     matmul::Strategy::PlaneMma,
        // );
        run::<R, half::f16>(Default::default(), matmul::Strategy::default());
        // run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::PlaneMma);
        // run::<cubecl::cuda::CudaRuntime, half::f16>(Default::default(), matmul::Strategy::PlaneMma);
        // run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::Accelerated);
        run::<R, half::f16>(Default::default(), matmul::Strategy::Accelerated);
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
        run::<cubecl::hip::HipRuntime, f32>(Default::default(), matmul::Strategy::Accelerated);
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
        run::<cubecl::hip::HipRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::Accelerated,
        );
    }

    #[cfg(feature = "cuda")]
    {
        // run::<cubecl::cuda::CudaRuntime, f32>(
        //     Default::default(),
        //     matmul::Strategy::Tiling2D(Default::default()),
        // );
        // run::<cubecl::cuda::CudaRuntime, half::f16>(
        //     Default::default(),
        //     matmul::Strategy::Tiling2D(Default::default()),
        // );
        // run::<cubecl::cuda::CudaRuntime, f32>(
        //     Default::default(),
        //     matmul::Strategy::CmmaOld(Default::default()),
        // );
        run::<cubecl::cuda::CudaRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::default(),
        );
        // run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::PlaneMma);
        // run::<cubecl::cuda::CudaRuntime, half::f16>(Default::default(), matmul::Strategy::PlaneMma);
        // run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), matmul::Strategy::Accelerated);
        run::<cubecl::cuda::CudaRuntime, half::f16>(
            Default::default(),
            matmul::Strategy::Accelerated,
        );
    }
}
