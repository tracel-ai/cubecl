use std::marker::PhantomData;

use crate::components::batch::partition_batch_matmul::{
    GlobalPartitionMatmul, PartitionRangeDim, PartitionRanges,
};
use crate::components::global::GlobalMatmulFamily;
use crate::components::global::Quantization;
use crate::components::{
    Args, EA, EI, EO, ES, InputRuntimeArg, InvalidConfigError, MatmulLineSizes, MatmulPrecision,
    MatmulProblem, MatmulSpec, OutputRuntimeArg,
};
use crate::components::{MatmulConfigFactory, MatmulLaunch, batch, config::MatmulConfig, global};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::matmul::MatmulSelection;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::{BatchConfig as _, BatchMatmulFamily, Partitioner};

pub struct PartitionedBatchMatmulFamily<
    GMM: GlobalMatmulFamily,
    S: GlobalPartitionMatmul,
    P: Partitioner,
> {
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<P>,
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, P: Partitioner> BatchMatmulFamily
    for PartitionedBatchMatmulFamily<GMM, S, P>
{
    type Matmul<MP: MatmulPrecision> = PartitionedBatchMatmul<MP, GMM::Matmul<MP>, S, P>;

    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let elements_in_m = selection.tiling_scheme.elements_in_global_partition_m();
        let elements_in_n = selection.tiling_scheme.elements_in_global_partition_n();

        P::create_cube_count(
            (problem.m as u32).div_ceil(elements_in_m),
            (problem.n as u32).div_ceil(elements_in_n),
            (problem.num_batches() as u32)
                .div_ceil(selection.tiling_scheme.global_partition_size.batches),
        )
    }
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, C: Partitioner> MatmulConfigFactory
    for PartitionedBatchMatmulFamily<GMM, S, C>
{
    type Config = Config<GMM::Config, C>;
    type Input = GMM::Input;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        GMM::check_config(&config.global_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        GMM::check_availability::<R, MP>(client, &config.global_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let global_config =
            GMM::make_config(input, problem, line_sizes, cube_dim, cube_count, quantized);
        let cube_count = if let CubeCount::Static(x, y, z) = cube_count {
            (*x, *y, *z)
        } else {
            panic!("Dynamic cube count unsupported")
        };

        Config::new(global_config, cube_count, quantized)
    }
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, C: Partitioner> MatmulLaunch
    for PartitionedBatchMatmulFamily<GMM, S, C>
{
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: Self::Config,
    ) {
        unsafe {
            super::matmul::launch_unchecked::<Args<MS>, EI<MS>, ES<MS>, EA<MS>, EO<MS>, Self, R>(
                client, cube_count, cube_dim, input, output, config,
            );
        }
    }
}

/// Executes matrix multiplication at the batch level,
/// assigning each cube to handle multiple global matmuls.
///
/// The algorithm supports any number of cubes,
/// looping as needed to process all data.
pub struct PartitionedBatchMatmul<
    MP: MatmulPrecision,
    GMM: global::GlobalMatmul<MP>,
    S: GlobalPartitionMatmul,
    P: Partitioner,
> {
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<P>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    GMM: global::GlobalMatmul<MP>,
    GPMM: GlobalPartitionMatmul,
    P: Partitioner,
> batch::BatchMatmul<MP> for PartitionedBatchMatmul<MP, GMM, GPMM, P>
{
    type Config = Config<GMM::Config, P>;

    fn execute(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) {
        let rank = out.rank();
        let problem_m = out.shape(rank - 2);
        let problem_n = out.shape(rank - 1);
        let problem_k = lhs.shape(lhs.rank() - 1);
        let k_range = (0, problem_k);

        let mut problem_b = 1;
        for b in 0..rank - 2 {
            problem_b *= out.shape(b);
        }

        let ts = config.tiling_scheme();
        let (m_index, n_index) = P::m_n_indices();
        let batch_index = P::batch_index();

        let ranges = PartitionRanges::new(
            PartitionRangeDim::new(
                problem_m,
                m_index,
                ts.elements_in_stage_m(),
                ts.elements_in_global_partition_m(),
            ),
            PartitionRangeDim::new(
                problem_n,
                n_index,
                ts.elements_in_stage_n(),
                ts.elements_in_global_partition_n(),
            ),
            PartitionRangeDim::new(
                problem_b,
                batch_index,
                1u32,
                ts.global_partition_size.batches,
            ),
        );

        let global_config = config.global_config();
        let acc = GMM::init_accumulator(global_config);

        GPMM::execute::<MP, GMM>(
            lhs,
            rhs,
            out,
            ranges,
            acc,
            k_range,
            quantization,
            global_config,
        );
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct Config<G: global::GlobalConfig, P: Partitioner> {
    global_config: G,
    cube_count: (u32, u32, u32),
    quantized: bool,
    _c: PhantomData<P>,
}

impl<G: global::GlobalConfig, P: Partitioner> batch::BatchConfig for Config<G, P> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn max_problem_m(&self) -> u32 {
        self.cube_count_m()
            * self
                .global_config
                .tiling_scheme()
                .elements_in_global_partition_m()
    }

    fn max_problem_n(&self) -> u32 {
        self.cube_count_n()
            * self
                .global_config
                .tiling_scheme()
                .elements_in_global_partition_n()
    }

    fn max_problem_batches(&self) -> u32 {
        self.cube_count_batch()
            * self
                .global_config
                .tiling_scheme()
                .global_partition_size
                .batches
    }

    fn quantized(&self) -> bool {
        self.quantized
    }
}

impl<G: global::GlobalConfig, P: Partitioner> MatmulConfig for Config<G, P> {}

impl<G: global::GlobalConfig, P: Partitioner> Config<G, P> {
    pub fn new(global_config: G, cube_count: (u32, u32, u32), quantized: bool) -> Self {
        Self {
            global_config,
            cube_count,
            quantized,
            _c: PhantomData,
        }
    }

    fn cube_count_m(&self) -> u32 {
        P::cube_count_m(self.cube_count)
    }

    fn cube_count_n(&self) -> u32 {
        P::cube_count_n(self.cube_count)
    }

    fn cube_count_batch(&self) -> u32 {
        P::cube_count_batches(self.cube_count)
    }
}
