use core::marker::PhantomData;
use cubecl::{Feature, TmaFeature, prelude::*};
use cubecl_matmul::components::stage::PartitionBuffering;
use cubecl_matmul::components::{
    LoadSpecializationConfig, SpecializationTensorConfig, TilingScheme,
};
use cubecl_matmul::kernels::matmul::double_buffering::DoubleBufferingArgs;
use cubecl_matmul::kernels::matmul::ordered_double_buffering::OrderedSelectionArgs;
use cubecl_matmul::kernels::matmul::simple::SimpleArgs;
use cubecl_matmul::kernels::matmul::{
    MatmulSelection, MultiRowStrategy, Selection, closest_factor_pair,
};
use cubecl_matmul::{self as matmul};
use cubecl_matmul::{AsyncLoadingStrategy, components::MatmulPrecision};
use cubecl_matmul::{SyncBufferLoadingStrategy, SyncLoadingStrategy};
use std::time::Duration;

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl_runtime::config::GlobalConfig;
use cubecl_std::tensor::TensorHandle;

use cubecl_random::random_uniform;

impl<R: Runtime, MP: MatmulPrecision> Benchmark for MatmulBench<R, MP> {
    type Output = ();
    type Input = (
        TensorHandle<R, MP::EI>,
        Option<TensorHandle<R, f32>>,
        TensorHandle<R, MP::EI>,
        Option<TensorHandle<R, f32>>,
    );

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let mut lhs = TensorHandle::<R, MP::EI>::empty(&client, vec![self.b, self.m, self.k]);
        if self.tl {
            let len = lhs.shape.len();
            lhs.strides.swap(len - 2, len - 1);
        }
        random_uniform::<R, MP::EI>(
            &client,
            MP::EI::from_int(0),
            MP::EI::from_int(1),
            lhs.as_ref(),
        );

        let mut rhs = TensorHandle::<R, MP::EI>::empty(&client, vec![self.b, self.k, self.n]);

        if self.tr {
            let len = rhs.shape.len();
            rhs.strides.swap(len - 2, len - 1);
        }

        random_uniform::<R, MP::EI>(
            &client,
            MP::EI::from_int(0),
            MP::EI::from_int(1),
            rhs.as_ref(),
        );

        (lhs, None, rhs, None)
    }

    fn execute(&self, (lhs, lhs_scale, rhs, rhs_scale): Self::Input) -> Self::Output {
        let client = R::client(&self.device);
        let out = TensorHandle::empty(&client, vec![self.b, self.m, self.n]);

        match matmul::launch::<R, MP>(
            &self.strategy,
            &self.client,
            lhs,
            lhs_scale,
            rhs,
            rhs_scale,
            out,
        ) {
            Ok(_) => return (),
            Err(err) => {
                println!("{err:?}");
            }
        }
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

    fn profile(&self, args: Self::Input) -> cubecl::benchmark::ProfileDuration {
        self.client.profile(|| self.execute(args)).unwrap()
    }
}

#[allow(dead_code)]
struct MatmulBench<R: Runtime, MP> {
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    tl: bool,
    tr: bool,
    strategy: matmul::Strategy,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _mp: PhantomData<MP>,
}

#[allow(dead_code)]
fn run<R: Runtime, MP: MatmulPrecision>(device: R::Device, strategy: matmul::Strategy) {
    let client = R::client(&device);

    // for tl in [true, false] {
    // for tr in [true, false] {
    for tl in [false] {
        for tr in [false] {
            for (b, m, n, k) in [
                // (1, 8192, 8192, 8192),
                (1, 6144, 6144, 6144),
                // (1, 5000, 5000, 5000),
                // (2, 4096, 4096, 4096),
                // (5, 512, 512, 512),
                // (10, 256, 256, 256),
                // OuterProduct
                // (2, 4096, 4096, 1),
                (1, 4 * 4096, 4 * 4096, 1),
                // InnerProduct
                // (2, 1, 8 * 4096, 1),
                // VecScalar
                // (2, 8 * 4096, 1, 1),
                // ScalarVec
                // (2, 1, 4096, 1),
                // MatVec
                // (2, 4096, 1, 4096),
                // VecMat
                // (2, 1, 4096, 4096),
                // General
                // (2, 4096, 4096, 4096),
            ] {
                let bench = MatmulBench::<R, MP> {
                    b,
                    m,
                    k,
                    n,
                    tl,
                    tr,
                    client: client.clone(),
                    device: device.clone(),
                    strategy: strategy.clone(),
                    _mp: PhantomData,
                };
                println!("b: {b} m: {m} n: {n} k: {k}, tl {tl}, tr {tr}");
                println!("{}", bench.name());
                println!("{}", bench.run(TimingMethod::System));
            }
        }
    }
}

#[allow(unused)]
fn run_benches<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    println!("Simple");
    run::<R, MP>(Default::default(), matmul::Strategy::SimpleUnit(None));

    // println!("Simple multi rows");
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::Simple(
    //         SyncLoadingStrategy::Cyclic,
    //         Selection::Inferred(SimpleArgs { multi_rows: true }),
    //     ),
    // );

    // println!("Double Buffering");
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::DoubleBuffering(
    //         SyncBufferLoadingStrategy::Tilewise,
    //         Selection::Inferred(DoubleBufferingArgs { specialized: false }),
    //     ),
    // );

    // println!("Double Buffering Specialized");
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::DoubleBuffering(
    //         SyncBufferLoadingStrategy::Tilewise,
    //         Selection::Inferred(DoubleBufferingArgs { specialized: true }),
    //     ),
    // );

    // println!("Ordered 1");
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::OrderedDoubleBuffering(Selection::Inferred(OrderedSelectionArgs {
    //         row_count: Some(16),
    //         rows_per_plane: Some(2),
    //         partition_k: Some(1),
    //     })),
    // );

    // println!("Ordered 2");
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::OrderedDoubleBuffering(Selection::Inferred(OrderedSelectionArgs {
    //         row_count: Some(8),
    //         rows_per_plane: Some(2),
    //         partition_k: Some(2),
    //     })),
    // );
}

fn main() {
    #[cfg(all(
        feature = "wgpu",
        not(feature = "wgpu-spirv"),
        not(feature = "wgpu-msl")
    ))]
    {
        run_benches::<cubecl::wgpu::WgpuRuntime, f32>();
        // run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
    }

    #[cfg(feature = "wgpu-spirv")]
    {
        // run_benches::<cubecl::wgpu::WgpuRuntime, f32>();
        run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
    }

    #[cfg(all(feature = "hip", target_os = "linux"))]
    {
        // run_benches::<cubecl::hip::HipRuntime, f32>();
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
        run_benches::<cubecl::wgpu::WgpuRuntime, f32>();
    }
}
