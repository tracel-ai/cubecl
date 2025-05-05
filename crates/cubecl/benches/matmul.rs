use core::marker::PhantomData;
use cubecl::{Feature, TmaFeature, prelude::*};
use cubecl_linalg::matmul::SyncBufferLoadingStrategy;
use cubecl_linalg::matmul::{self, AsyncLoadingStrategy, components::MatmulPrecision};

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl_linalg::tensor::TensorHandle;
use cubecl_runtime::config::GlobalConfig;

impl<R: Runtime, MP: MatmulPrecision> Benchmark for MatmulBench<R, MP> {
    type Args = (TensorHandle<R, MP::EI>, TensorHandle<R, MP::EI>);

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);

        let lhs = TensorHandle::zeros(&client, vec![self.b, self.m, self.k]);
        let rhs = TensorHandle::zeros(&client, vec![self.b, self.k, self.n]);

        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        let client = R::client(&self.device);
        let out = TensorHandle::empty(&client, vec![self.b, self.m, self.n]);

        matmul::launch::<R, MP>(&self.strategy, &self.client, lhs, rhs, out).unwrap();
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);

        format!(
            "{}-matmul{}-{}-{}-{}-{}-{:?}",
            R::name(&client),
            if MP::QUANTIZED { "-quantized" } else { "" },
            MP::EI::as_elem_native_unchecked(),
            MP::ES::as_elem_native_unchecked(),
            MP::EA::as_elem_native_unchecked(),
            MP::EO::as_elem_native_unchecked(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        GlobalConfig::get();
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Args) -> cubecl::benchmark::ProfileDuration {
        self.client.profile(|| self.execute(args))
    }
}

#[allow(dead_code)]
struct MatmulBench<R: Runtime, MP> {
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    strategy: matmul::Strategy,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _mp: PhantomData<MP>,
}

#[allow(dead_code)]
fn run<R: Runtime, MP: MatmulPrecision>(device: R::Device, strategy: matmul::Strategy) {
    let client = R::client(&device);

    for (b, m, n, k) in [
        (1, 8192, 8192, 8192),
        (1, 6144, 6144, 6144),
        (1, 5000, 5000, 5000),
        (2, 4096, 4096, 4096),
        (32, 1024, 1024, 1024),
    ] {
        let bench = MatmulBench::<R, MP> {
            b,
            m,
            k,
            n,
            client: client.clone(),
            device: device.clone(),
            strategy: strategy.clone(),
            _mp: PhantomData,
        };
        println!("b: {b} m: {m} n: {n} k: {k}");
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::Full));
    }
}

#[allow(unused)]
fn run_benches<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleBuffering(SyncBufferLoadingStrategy::Tilewise),
    );
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleBuffering(SyncBufferLoadingStrategy::Cyclic),
    );
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleBuffering(SyncBufferLoadingStrategy::Hybrid),
    );
    // // run::<R, MP>(
    // //     Default::default(),
    // //     matmul::Strategy::Simple(SyncLoadingStrategy::Strided),
    // // );
    // // run::<R, MP>(
    // //     Default::default(),
    // //     matmul::Strategy::SimpleBarrier(AsyncLoadingStrategy::Cyclic),
    // // );
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::Tiling2D(Default::default()),
    // );
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::SimpleBarrier(AsyncLoadingStrategy::Cooperative),
    // );

    if client
        .properties()
        .feature_enabled(Feature::Tma(TmaFeature::Base))
    {
        run::<R, MP>(
            Default::default(),
            matmul::Strategy::SimpleBarrier(AsyncLoadingStrategy::Tma),
        );
    }
}

fn main() {
    #[cfg(feature = "wgpu")]
    {
        // run_benches::<cubecl::wgpu::WgpuRuntime, f32>();
    }

    #[cfg(feature = "wgpu-spirv")]
    {
        // run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
        run_benches::<cubecl::wgpu::WgpuRuntime, cubecl::flex32>();
    }

    #[cfg(all(feature = "hip", target_os = "linux"))]
    {
        run_benches::<cubecl::hip::HipRuntime, half::f16>();
    }

    #[cfg(feature = "cuda")]
    {
        // run_benches::<cubecl::cuda::CudaRuntime, f32>();
        run_benches::<cubecl::cuda::CudaRuntime, half::f16>();
        // run_benches::<cubecl::cuda::CudaRuntime, cubecl::flex32>();
        // run_benches::<cubecl::cuda::CudaRuntime, cubecl_std::SymQ8>();
        // run_benches::<cubecl::cuda::CudaRuntime, (i8, i8, i32, i32)>();
        // run_benches::<cubecl::cuda::CudaRuntime, (i8, i8, i32, i8)>();
        // run_benches::<cubecl::cuda::CudaRuntime, (i8, half::f16, half::f16, half::f16)>();
        // run_benches::<cubecl::cuda::CudaRuntime, (i8, half::bf16, f32, f32)>();
        // run_benches::<cubecl::cuda::CudaRuntime, (i8, half::f16, f32, half::f16)>();
    }
    #[cfg(feature = "wgpu-msl")]
    {
        run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
    }
}
