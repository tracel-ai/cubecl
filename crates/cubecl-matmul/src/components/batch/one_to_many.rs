use std::marker::PhantomData;

use crate::components::batch::span::{GlobalPartitionMatmul, PartitionSpan, SpanDim};
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

use super::{BatchConfig as _, BatchMatmulFamily, CubeDispatch};

pub struct OneToManyMatmulFamily<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, C: CubeDispatch>
{
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<C>,
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, C: CubeDispatch> BatchMatmulFamily
    for OneToManyMatmulFamily<GMM, S, C>
{
    type Matmul<MP: MatmulPrecision> = OneToManyMatmul<MP, GMM::Matmul<MP>, S, C>;

    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let elements_in_m = selection.tiling_scheme.elements_in_stage_m()
            * selection.tiling_scheme.global_partition_size.m;
        let elements_in_n = selection.tiling_scheme.elements_in_stage_n()
            * selection.tiling_scheme.global_partition_size.n;

        C::cube_count(
            (problem.m as u32).div_ceil(elements_in_m),
            (problem.n as u32).div_ceil(elements_in_n),
            (problem.num_batches() as u32)
                .div_ceil(selection.tiling_scheme.global_partition_size.batches),
        )
    }
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, C: CubeDispatch> MatmulConfigFactory
    for OneToManyMatmulFamily<GMM, S, C>
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

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, C: CubeDispatch> MatmulLaunch
    for OneToManyMatmulFamily<GMM, S, C>
{
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        size_k: ScalarArg<u32>,
        config: Self::Config,
    ) {
        unsafe {
            super::matmul::launch_unchecked::<Args<MS>, EI<MS>, ES<MS>, EA<MS>, EO<MS>, Self, R>(
                client, cube_count, cube_dim, input, output, size_k, config,
            );
        }
    }
}

/// Executes matrix multiplication at the batch level,
/// assigning each cube to handle multiple global matmuls.
///
/// The algorithm supports any number of cubes,
/// looping as needed to process all data.
pub struct OneToManyMatmul<
    MP: MatmulPrecision,
    GMM: global::GlobalMatmul<MP>,
    S: GlobalPartitionMatmul,
    C: CubeDispatch,
> {
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<C>,
}

#[cube]
impl<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>, S: GlobalPartitionMatmul, C: CubeDispatch>
    batch::BatchMatmul<MP> for OneToManyMatmul<MP, GMM, S, C>
{
    type Config = Config<GMM::Config, C>;

    fn execute(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        _size_k: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) {
        let rank = out.rank();
        let problem_m = out.shape(rank - 2);
        let problem_n = out.shape(rank - 1);

        let mut shape_b = 1;
        for b in 0..rank - 2 {
            shape_b *= out.shape(b);
        }

        let cubes_m = config.cube_count_m();
        let cubes_n = config.cube_count_n();
        let cubes_b = config.cube_count_batch();

        let stage_m = config.tiling_scheme().elements_in_stage_m();
        let stage_n = config.tiling_scheme().elements_in_stage_n();
        let stage_b = 1;

        let (m_index, n_index) = C::m_n_indices();
        let batch_index = C::batch_index();

        let span = PartitionSpan::new(
            SpanDim::new(problem_m, stage_m, m_index, cubes_m),
            SpanDim::new(problem_n, stage_n, n_index, cubes_n),
            SpanDim::new(shape_b, stage_b, batch_index, cubes_b),
        );

        let k_range = (0, lhs.shape(rank - 1));

        let global_config = config.global_config();
        let acc = GMM::init_accumulator(global_config);
        S::execute::<MP, GMM>(
            lhs,
            rhs,
            out,
            span,
            acc,
            k_range,
            quantization,
            global_config,
        );
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct Config<G: global::GlobalConfig, C: CubeDispatch> {
    global_config: G,
    cube_count: (u32, u32, u32),
    quantized: bool,
    _c: PhantomData<C>,
}

impl<G: global::GlobalConfig, C: CubeDispatch> batch::BatchConfig for Config<G, C> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn max_m(&self) -> u32 {
        u32::maximum_value()
    }

    fn max_n(&self) -> u32 {
        u32::maximum_value()
    }

    fn max_batches(&self) -> u32 {
        u32::maximum_value()
    }

    fn quantized(&self) -> bool {
        self.quantized
    }
}

impl<G: global::GlobalConfig, C: CubeDispatch> MatmulConfig for Config<G, C> {}

impl<G: global::GlobalConfig, C: CubeDispatch> Config<G, C> {
    pub fn new(global_config: G, cube_count: (u32, u32, u32), quantized: bool) -> Self {
        Self {
            global_config,
            cube_count,
            quantized,
            _c: PhantomData,
        }
    }

    fn cube_count_m(&self) -> u32 {
        C::max_m(self.cube_count)
    }

    fn cube_count_n(&self) -> u32 {
        C::max_n(self.cube_count)
    }

    fn cube_count_batch(&self) -> u32 {
        C::max_batches(self.cube_count)
    }
}
