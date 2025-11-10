use cubecl::prelude::*;
use cubecl_matmul::AcceleratedTileKind;
use cubecl_matmul::components::batch::HypercubeSelection;
use cubecl_matmul::components::stage::PartitionBuffering;
use cubecl_matmul::components::{
    LoadingPrecomputeStrategy, MatmulElems, MatmulPrecision, MatmulSelection, StageSize,
    TilingScheme,
};
use cubecl_matmul::kernels::layered::double_buffering::DoubleBufferingArgs;
use cubecl_matmul::kernels::layered::double_unit::DoubleUnitSelectionArgs;
use cubecl_matmul::kernels::layered::ordered_double_buffering::OrderedSelectionArgs;
use cubecl_matmul::kernels::layered::simple::SimpleArgs;
use cubecl_matmul::kernels::layered::simple_unit::SimpleUnitSelectionArgs;
use cubecl_matmul::kernels::layered::{Selection, TileSizeSelection};
use cubecl_matmul::{self as matmul, MatmulInputHandle, PartialReadingStrategy, ReadingStrategy};
use std::collections::BTreeMap;

use cubecl::benchmark::{Benchmark, BenchmarkComputations, BenchmarkDurations, TimingMethod};
use cubecl::future;
use cubecl_runtime::config::GlobalConfig;
use cubecl_std::tensor::TensorHandle;

use cubecl_random::random_uniform;

impl<R: Runtime> Benchmark for MatmulBench<R> {
    type Input = (MatmulInputHandle<R>, MatmulInputHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let mut lhs = TensorHandle::<R>::empty(
            &client,
            vec![self.b, self.m, self.k],
            self.dtypes.lhs_global,
        );
        if self.tl {
            let len = lhs.shape.len();
            lhs.strides.swap(len - 2, len - 1);
        }
        random_uniform::<R>(&client, 0.0, 1.0, lhs.as_ref(), self.dtypes.lhs_global);

        let mut rhs = TensorHandle::<R>::empty(
            &client,
            vec![self.b, self.k, self.n],
            self.dtypes.rhs_global,
        );

        if self.tr {
            let len = rhs.shape.len();
            rhs.strides.swap(len - 2, len - 1);
        }

        random_uniform::<R>(&client, 0.0, 1.1, rhs.as_ref(), self.dtypes.rhs_global);

        (
            MatmulInputHandle::Normal(lhs),
            MatmulInputHandle::Normal(rhs),
        )
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, String> {
        let client = R::client(&self.device);
        let out = TensorHandle::empty(
            &client,
            vec![self.b, self.m, self.n],
            self.dtypes.acc_global,
        );

        match matmul::launch::<R>(
            &self.strategy,
            &self.client,
            lhs,
            rhs,
            out,
            self.dtypes.clone(),
        ) {
            Ok(_) => Ok(()),
            Err(err) => Err(format!("{err:?}")),
        }
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);

        format!(
            "{}-matmul-Lhs<{}-{}-{}>-Rhs<{}-{}-{}>-{}-{}-{:?}",
            R::name(&client),
            self.dtypes.lhs_global,
            self.dtypes.lhs_stage,
            self.dtypes.lhs_register,
            self.dtypes.rhs_global,
            self.dtypes.rhs_stage,
            self.dtypes.rhs_register,
            self.dtypes.acc_register,
            self.dtypes.acc_global,
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
struct MatmulBench<R: Runtime> {
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    tl: bool,
    tr: bool,
    strategy: matmul::Strategy,
    device: R::Device,
    client: ComputeClient<R::Server>,
    dtypes: MatmulElems,
}

#[allow(unused)]
fn entry(m: usize, n: usize, k: usize) -> (usize, usize, usize, usize) {
    let expected = 2 * 6144 * 6144 * 6144;
    let num_ops = 2 * m * n * k;

    let b = usize::max(expected / num_ops, 1);
    let b = 2usize.pow((b as f64).log(2.0).floor() as u32);
    let b = usize::min(4096, b);

    (b, m, n, k)
}

#[allow(dead_code, clippy::single_element_loop)]
fn run<R: Runtime, MP: MatmulPrecision>(device: R::Device, strategy: matmul::Strategy) {
    for tl in [true, false] {
        for tr in [true, false] {
            for (b, m, n, k) in [
                // entry(8192, 8192, 8192),
                // entry(6144, 6144, 6144),
                // entry(4096, 4096, 4096),
                // entry(2048, 2048, 2048),
                // (2, 1024, 1024, 1024),
                // entry(512, 512, 512),
                // entry(64, 1024, 64),
                // entry(32, 1024, 32),
                // entry(10, 1024, 10),
                // entry(64, 64, 1024),
                // entry(32, 32, 1024),
                // entry(10, 10, 1024),
                // entry(1024, 64, 64),
                // entry(1024, 32, 32),
                // entry(1024, 10, 10),
                // (16, 1, 2048, 8192),
                // (16, 1, 4096, 4096),
                // (16, 1, 512, 4096),
                // (2, 8192, 8192, 1), // Outer
                // (2, 8192, 1, 8192), // MatVec
                (2, 1, 8192, 8192), // VecMat
            ] {
                println!("-------------------");
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

    let bench = MatmulBench::<R> {
        b,
        m,
        k,
        n,
        tl,
        tr,
        client: client.clone(),
        device: device.clone(),
        strategy: strategy.clone(),
        dtypes: MatmulElems::new::<MP>(),
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

#[allow(unused, clippy::single_element_loop)]
// This function should be customized to help build a proper selector that reduces the number of
// possibilities.
fn run_grid_search<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    let mut algos = BTreeMap::<u64, (BenchmarkDurations, MatmulSelection, f64)>::new();

    for t in [(16, 16, 16)] {
        for p in [(1, 1, 1)] {
            for s in [(1, 1, 1)] {
                let plane_dim = client.properties().hardware.plane_size_min;
                let tiling = TilingScheme::builder()
                    .with_tile_size(t.into())
                    .with_partition_size(p.into())
                    .with_stage_size(StageSize {
                        m: s.0,
                        n: s.1,
                        k: s.2,
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
                    .partition_buffering(PartitionBuffering::Single)
                    .hypercube_config(hypercube)
                    .loading_precompute_strategy(LoadingPrecomputeStrategy::Always)
                    .build();
                let result = run_one::<R, MP>(
                    Default::default(),
                    matmul::Strategy::Simple {
                        read_strategy: ReadingStrategy::Cyclic,
                        selection: Selection::Forced(selection.clone()),
                        tile_kind: AcceleratedTileKind::Cmma,
                    },
                    (4096, 10, 64, 10),
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
fn run_algos_vecmat<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    println!("Simple VecMat");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::SimpleVecMat(Selection::Inferred(())),
    );

    println!("Double VecMat");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleVecMat(Selection::Inferred(())),
    );

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
        matmul::Strategy::Simple {
            read_strategy: ReadingStrategy::Cyclic,
            selection: Selection::Inferred(SimpleArgs { multi_rows: false }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Simple multi rows");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::Simple {
            read_strategy: ReadingStrategy::Cyclic,
            selection: Selection::Inferred(SimpleArgs { multi_rows: true }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Double Buffering");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleBuffering {
            read_strategy: PartialReadingStrategy::Tilewise,
            selection: Selection::Inferred(DoubleBufferingArgs { specialized: false }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Double Buffering Specialized");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::DoubleBuffering {
            read_strategy: PartialReadingStrategy::Tilewise,
            selection: Selection::Inferred(DoubleBufferingArgs { specialized: true }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Double Buffering Ordered");
    run::<R, MP>(
        Default::default(),
        matmul::Strategy::OrderedDoubleBuffering {
            selection: Selection::Inferred(OrderedSelectionArgs {
                row_count: Some(8),
                rows_per_plane: Some(2),
                partition_k: Some(2),
            }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );
}

#[allow(unused)]
fn run_benches<R: Runtime, MP: MatmulPrecision>() {
    // run_grid_search::<R, MP>();
    run_algos_unit::<R, MP>();
    run_algos_wmma::<R, MP>();
    // run_algos_vecmat::<R, MP>();
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
