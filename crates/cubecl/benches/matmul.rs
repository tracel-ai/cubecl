use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_matmul::components::batch::HypercubeSelection;
use cubecl_matmul::components::stage::PartitionBuffering;
use cubecl_matmul::components::{
    LoadingPrecomputeStrategy, MatmulPrecision, MatmulSelection, StageSize, TilingScheme,
};
use cubecl_matmul::kernels::layered::double_buffering::DoubleBufferingArgs;
use cubecl_matmul::kernels::layered::double_unit::DoubleUnitSelectionArgs;
use cubecl_matmul::kernels::layered::ordered_double_buffering::OrderedSelectionArgs;
use cubecl_matmul::kernels::layered::simple::SimpleArgs;
use cubecl_matmul::kernels::layered::simple_unit::SimpleUnitSelectionArgs;
use cubecl_matmul::kernels::layered::{Selection, TileSizeSelection};
use cubecl_matmul::{self as matmul, SyncLoadingStrategy, SyncPartialLoadingStrategy};
use std::collections::BTreeMap;

use cubecl::benchmark::{Benchmark, BenchmarkComputations, BenchmarkDurations, TimingMethod};
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

    fn execute(
        &self,
        (lhs, lhs_scale, rhs, rhs_scale): Self::Input,
    ) -> Result<Self::Output, String> {
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
            Ok(_) => Ok(()),
            Err(err) => Err(format!("{err:?}")),
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

    fn profile(&self, args: Self::Input) -> Result<cubecl::benchmark::ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "matmul-bench")
            .map_err(|err| format!("{err:?}"))
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

fn entry(m: usize, n: usize, k: usize) -> (usize, usize, usize, usize) {
    let expected = 2 * 6144 * 6144 * 6144;
    let num_ops = 2 * m * n * k;

    let b = usize::max(expected / num_ops, 1);
    let b = 2usize.pow((b as f64).log(2.0).floor() as u32);

    (b, m, n, k)
}

#[allow(dead_code)]
fn run<R: Runtime, MP: MatmulPrecision>(device: R::Device, strategy: matmul::Strategy) {
    for tl in [false] {
        for tr in [false] {
            for (b, m, n, k) in [
                entry(8192, 8192, 8192),
                entry(6144, 6144, 6144),
                entry(4096, 4096, 4096),
                entry(2048, 2048, 2048),
                entry(1024, 1024, 1024),
                entry(512, 512, 512),
            ] {
                let _ = run_one::<R, MP>(device.clone(), strategy.clone(), (b, m, n, k), (tl, tr));
            }
        }
    }
}

#[allow(dead_code)]
fn run_one<R: Runtime, MP: MatmulPrecision>(
    device: R::Device,
    strategy: matmul::Strategy,
    shapes: (usize, usize, usize, usize),
    transposed: (bool, bool),
) -> Result<(BenchmarkDurations, f64), String> {
    let client = R::client(&device);
    let (b, m, n, k) = shapes;
    let (tl, tr) = transposed;

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

    match bench.run(TimingMethod::System) {
        Ok(val) => {
            let flops = 2 * b * m * n * k;
            let computed = BenchmarkComputations::new(&val);
            let tflops = flops as f64 / (computed.median.as_secs_f64() * 1e12);
            println!("TFLOPS: {tflops}");
            println!("Times: {val}");
            Ok((val, tflops))
        }
        Err(err) => {
            println!("{err:?}");
            Err(err)
        }
    }
}

#[allow(unused)]
// This function should be customized to help build a proper selector that reduces the number of
// possibilities.
fn run_grid_search<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    let mut algos = BTreeMap::<u64, (BenchmarkDurations, MatmulSelection, f64)>::new();

    for t in [(1, 4, 4), (4, 4, 4)] {
        for p in [(1, 2, 8), (1, 2, 2), (8, 2, 2), (4, 4, 4)] {
            for s in [(8, 8), (16, 16)] {
                let plane_dim = client.properties().hardware.plane_size_min;
                let tiling = TilingScheme::builder()
                    .with_tile_size(t.into())
                    .with_partition_size(p.into())
                    .with_stage_size(StageSize {
                        m: s.0,
                        n: s.1,
                        k: 1,
                    })
                    .build()
                    .unwrap();
                let hypercube = HypercubeSelection::builder(&tiling)
                    .global_order(cubecl_matmul::components::batch::GlobalOrderSelection::Default)
                    .cube_count_plan(
                        cubecl_matmul::components::batch::CubeCountPlanSelection::Flattened,
                    )
                    .build();
                let selection = MatmulSelection::builder(tiling, plane_dim)
                    .plane_dim(plane_dim)
                    .partition_buffering(PartitionBuffering::Double)
                    .hypercube_config(hypercube)
                    .loading_precompute_strategy(LoadingPrecomputeStrategy::Always)
                    .build();
                let result = run_one::<R, MP>(
                    Default::default(),
                    matmul::Strategy::DoubleUnit(Selection::Forced(selection.clone())),
                    // (8, 1024, 1024, 1024),
                    (1, 4096, 4096, 4096),
                    (false, false),
                );

                if let Ok((duration, tflops)) = result {
                    let key = tflops * 1000000.0;
                    algos.insert(key as u64, (duration, selection, tflops));
                }
            }
        }
    }

    for (_, (duration, selection, tflops)) in algos.iter() {
        println!("==== TFLOPS: {tflops:?} ====");
        println!("Selection: {selection:?}");
        println!("Times: {duration}");
        println!("====================");
    }
}

#[allow(unused)]
fn run_algos_unit<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    println!("Simple Unit Min");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
    );

    println!("Simple Unit Max");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
    );

    println!("Double Unit Min");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleUnit(Selection::Inferred(DoubleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
    );
    println!("Double Unit Max");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleUnit(Selection::Inferred(DoubleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
    );
}

#[allow(unused)]
fn run_algos_wmma<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    println!("Simple");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::Simple(
            SyncLoadingStrategy::Cyclic,
            Selection::Inferred(SimpleArgs { multi_rows: false }),
        ),
    );

    println!("Simple multi rows");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::Simple(
            SyncLoadingStrategy::Cyclic,
            Selection::Inferred(SimpleArgs { multi_rows: true }),
        ),
    );

    println!("Double Buffering");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleBuffering(
            SyncPartialLoadingStrategy::Tilewise,
            Selection::Inferred(DoubleBufferingArgs { specialized: false }),
        ),
    );

    println!("Double Buffering Specialized");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleBuffering(
            SyncPartialLoadingStrategy::Tilewise,
            Selection::Inferred(DoubleBufferingArgs { specialized: true }),
        ),
    );

    println!("Double Buffering Ordered");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::OrderedDoubleBuffering(Selection::Inferred(OrderedSelectionArgs {
            row_count: Some(8),
            rows_per_plane: Some(2),
            partition_k: Some(2),
        })),
    );
}

#[allow(unused)]
fn run_benches<R: Runtime, MP: MatmulPrecision>() {
    // run_grid_search::<R, MP>();
    // run_algos_unit::<R, MP>();
    run_algos_wmma::<R, MP>();
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
        run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
    }
}
