use std::marker::PhantomData;

use crate::matmul::components::batch::slice::{Slice, SliceDim, SliceMatmul};
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{
    batch, config::MatmulConfig, global, Ident, MatmulKernel, MatmulLaunch, StageDim,
};
use crate::matmul::kernels::matmul::AdvancedConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::Config as _;

/// Performs matrix multiplication at the batch level,
/// with one cube assigned to several underlying global matmuls
pub struct Matmul<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>, S: SliceMatmul> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>, S: SliceMatmul> batch::Matmul<EG>
    for Matmul<EG, ES, GMM, S>
{
    fn execute(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
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
        let cubes_z = config.cube_count_z();

        let stage_x = config.stage_dim(Ident::Out).num_elements_x_dim();
        let stage_y = config.stage_dim(Ident::Out).num_elements_y_dim();
        let stage_z = 1;

        let span = Slice::new(
            SliceDim::new(shape_x, stage_x, CUBE_POS_X, cubes_x),
            SliceDim::new(shape_y, stage_y, CUBE_POS_Y, cubes_y),
            SliceDim::new(shape_z, stage_z, CUBE_POS_Z, cubes_z),
        );

        let k_range = (0, lhs.shape(rank - 1));

        let gmm_config = config.to_gmm_config();
        let acc = GMM::init_accumulator(gmm_config);
        S::execute::<EG, ES, GMM>(lhs, rhs, out, span, acc, k_range, gmm_config);
    }
}

impl<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>, S: SliceMatmul> MatmulKernel<EG, EG>
    for Matmul<EG, ES, GMM, S>
{
    type Config = Config<GMM::Config>;

    fn check_config(config: Self::Config) {
        GMM::check_config(config.to_gmm_config())
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), &str> {
        GMM::check_availability::<R>(client)
    }

    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let gmm_config = GMM::make_config(problem, cube_dim, cube_count, advanced_config);
        let (cube_count_x, cube_count_y, cube_count_z) =
            if let CubeCount::Static(x, y, z) = cube_count {
                (x, y, z)
            } else {
                panic!("Dynamic cube count unsupported")
            };

        Config::new(gmm_config, *cube_count_x, *cube_count_y, *cube_count_z)
    }
}

impl<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>, S: SliceMatmul> MatmulLaunch<EG, EG>
    for Matmul<EG, ES, GMM, S>
{
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: Self::Config,
    ) {
        Self::check_config(config);
        super::launch::launch_unchecked::<EG, Self, R>(
            client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct Config<G: global::Config> {
    gmm_config: G,
    cube_count_x: u32,
    cube_count_y: u32,
    cube_count_z: u32,
}

impl<G: global::Config> batch::Config for Config<G> {
    type GmmConfig = G;

    fn to_gmm_config(&self) -> Self::GmmConfig {
        self.gmm_config
    }

    fn stage_dim(&self, ident: Ident) -> StageDim {
        self.gmm_config.stage_dim(ident)
    }

    fn cube_count_x(&self) -> u32 {
        self.cube_count_x
    }

    fn cube_count_y(&self) -> u32 {
        self.cube_count_y
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

impl<G: global::Config> MatmulConfig for Config<G> {}

impl<G: global::Config> Config<G> {
    pub fn new(gmm_config: G, cube_count_x: u32, cube_count_y: u32, cube_count_z: u32) -> Self {
        Self {
            gmm_config,
            cube_count_x,
            cube_count_y,
            cube_count_z,
        }
    }

    fn cube_count_z(&self) -> u32 {
        self.cube_count_z
    }
}
