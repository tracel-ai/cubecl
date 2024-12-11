use std::marker::PhantomData;

use crate::matmul::components::batch::span::{Span, SpanDim, SpanMatmul};
use crate::matmul::components::global::args::{TensorInput, TensorOutput};
use crate::matmul::components::{
    batch, config::MatmulConfig, global, Ident, MatmulKernel, MatmulLaunch, StageDim,
};
use crate::matmul::components::{InputRuntimeArg, MatmulProblem, MatmulSpec, OutputRuntimeArg};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Config as _, CubeDispatch};

/// Executes matrix multiplication at the batch level,
/// assigning each cube to handle multiple global matmuls.
///
/// The algorithm supports any number of cubes,
/// looping as needed to process all data.
pub struct Matmul<MS: MatmulSpec, GMM: global::Matmul<MS>, S: SpanMatmul, C: CubeDispatch> {
    _ms: PhantomData<MS>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<C>,
}

#[cube]
impl<MS: MatmulSpec, GMM: global::Matmul<MS>, S: SpanMatmul, C: CubeDispatch> batch::Matmul<MS>
    for Matmul<MS, GMM, S, C>
{
    fn execute(
        lhs: TensorInput<MS::EG, MS::Args>,
        rhs: TensorInput<MS::EG, MS::Args>,
        out: TensorOutput<MS::EG, MS::Args>,
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

        let stage_x = config.stage_dim(Ident::Out).num_elements_x_dim();
        let stage_y = config.stage_dim(Ident::Out).num_elements_y_dim();
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
        S::execute::<MS, GMM>(lhs, rhs, out, span, acc, k_range, gmm_config);
    }
}

impl<MS: MatmulSpec, GMM: global::Matmul<MS>, S: SpanMatmul, C: CubeDispatch> MatmulKernel
    for Matmul<MS, GMM, S, C>
{
    type Config = Config<GMM::Config, C>;

    fn check_config(config: Self::Config) {
        GMM::check_config(config.to_gmm_config())
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError> {
        GMM::check_availability::<R>(client)
    }

    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let gmm_config = GMM::make_config(problem, cube_dim, cube_count, advanced_config);
        let cube_count = if let CubeCount::Static(x, y, z) = cube_count {
            (*x, *y, *z)
        } else {
            panic!("Dynamic cube count unsupported")
        };

        Config::new(gmm_config, cube_count)
    }
}

impl<MS: MatmulSpec, GMM: global::Matmul<MS>, S: SpanMatmul, C: CubeDispatch> MatmulLaunch<MS>
    for Matmul<MS, GMM, S, C>
{
    unsafe fn launch_unchecked<'a, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: Self::Config,
    ) {
        Self::check_config(config);
        super::batch_matmul::launch_unchecked::<MS, Self, R>(
            client, cube_count, cube_dim, input, output, config,
        );
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct Config<G: global::Config, C: CubeDispatch> {
    gmm_config: G,
    cube_count: (u32, u32, u32),
    _c: PhantomData<C>,
}

impl<G: global::Config, C: CubeDispatch> batch::Config for Config<G, C> {
    type GmmConfig = G;

    fn to_gmm_config(&self) -> Self::GmmConfig {
        self.gmm_config
    }

    fn stage_dim(&self, ident: Ident) -> Box<dyn StageDim> {
        self.gmm_config.stage_dim(ident)
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

impl<G: global::Config, C: CubeDispatch> MatmulConfig for Config<G, C> {}

impl<G: global::Config, C: CubeDispatch> Config<G, C> {
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
