use std::marker::PhantomData;

use crate::matmul::components::batch::span::{Span, SpanDim, SpanMatmul};
use crate::matmul::components::global::GlobalMatmulFamily;
use crate::matmul::components::{
    batch, config::MatmulConfig, global, Ident, MatmulConfigFactory, MatmulLaunch, TilingDimensions,
};
use crate::matmul::components::{
    InputRuntimeArg, InvalidConfigError, MatmulPrecision, MatmulProblem, MatmulSpec,
    OutputRuntimeArg,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::{BatchConfig as _, BatchMatmulFamily, CubeDispatch};

pub struct OneToManyMatmulFamily<GMM: GlobalMatmulFamily, S: SpanMatmul, C: CubeDispatch> {
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<C>,
}

impl<GMM: GlobalMatmulFamily, S: SpanMatmul, C: CubeDispatch> BatchMatmulFamily
    for OneToManyMatmulFamily<GMM, S, C>
{
    type Matmul<MP: MatmulPrecision> = OneToManyMatmul<MP, GMM::Matmul<MP>, S, C>;
}

impl<GMM: GlobalMatmulFamily, S: SpanMatmul, C: CubeDispatch> MatmulConfigFactory
    for OneToManyMatmulFamily<GMM, S, C>
{
    type Config = Config<GMM::Config, C>;
    type Input = GMM::Input;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        GMM::check_config(&config.to_gmm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        GMM::check_availability::<R, MP>(client, &config.gmm_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let gmm_config = GMM::make_config(input, problem, cube_dim, cube_count, quantized);
        let cube_count = if let CubeCount::Static(x, y, z) = cube_count {
            (*x, *y, *z)
        } else {
            panic!("Dynamic cube count unsupported")
        };

        Config::new(gmm_config, cube_count)
    }
}

impl<GMM: GlobalMatmulFamily, S: SpanMatmul, C: CubeDispatch> MatmulLaunch
    for OneToManyMatmulFamily<GMM, S, C>
{
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: Self::Config,
    ) {
        super::matmul::launch_unchecked::<MS::EG, MS::ES, MS::EA, MS::Args, Self, R>(
            client, cube_count, cube_dim, input, output, config,
        );
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
    S: SpanMatmul,
    C: CubeDispatch,
> {
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<C>,
}

#[cube]
impl<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>, S: SpanMatmul, C: CubeDispatch>
    batch::BatchMatmul<MP> for OneToManyMatmul<MP, GMM, S, C>
{
    type Config = Config<GMM::Config, C>;

    fn execute(
        lhs: VirtualTensor<MP::EG>,
        rhs: VirtualTensor<MP::EG>,
        out: VirtualTensor<MP::EG, ReadWrite>,
        #[comptime] config: Self::Config,
    ) {
        let rank = out.rank();
        let shape_x = out.shape(rank - 2);
        let shape_y = out.shape(rank - 1);

        let mut shape_z = 1;
        for b in 0..rank - 2 {
            shape_z *= out.shape(b);
        }

        let cubes_x = config.cube_count_x();
        let cubes_y = config.cube_count_y();
        let cubes_z = config.cube_count_batch();

        let stage_x = config.tiling_dimensions(Ident::Out).total_row();
        let stage_y = config.tiling_dimensions(Ident::Out).total_col();
        let stage_z = 1;

        let (x_index, y_index) = C::x_y_indices();
        let batch_index = C::batch_index();

        let span = Span::new(
            SpanDim::new(shape_x, stage_x, x_index, cubes_x),
            SpanDim::new(shape_y, stage_y, y_index, cubes_y),
            SpanDim::new(shape_z, stage_z, batch_index, cubes_z),
        );

        let k_range = (0, lhs.shape(rank - 1));

        let gmm_config = config.to_gmm_config();
        let acc = GMM::init_accumulator(gmm_config);
        S::execute::<MP, GMM>(lhs, rhs, out, span, acc, k_range, gmm_config);
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct Config<G: global::GlobalConfig, C: CubeDispatch> {
    gmm_config: G,
    cube_count: (u32, u32, u32),
    _c: PhantomData<C>,
}

impl<G: global::GlobalConfig, C: CubeDispatch> batch::BatchConfig for Config<G, C> {
    type GmmConfig = G;

    fn to_gmm_config(&self) -> Self::GmmConfig {
        self.gmm_config
    }

    fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions {
        self.gmm_config.tiling_dimensions(ident)
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
}

impl<G: global::GlobalConfig, C: CubeDispatch> MatmulConfig for Config<G, C> {}

impl<G: global::GlobalConfig, C: CubeDispatch> Config<G, C> {
    pub fn new(gmm_config: G, cube_count: (u32, u32, u32)) -> Self {
        Self {
            gmm_config,
            cube_count,
            _c: PhantomData,
        }
    }

    fn cube_count_x(&self) -> u32 {
        C::max_x(self.cube_count)
    }

    fn cube_count_y(&self) -> u32 {
        C::max_y(self.cube_count)
    }

    fn cube_count_batch(&self) -> u32 {
        C::max_batches(self.cube_count)
    }
}
