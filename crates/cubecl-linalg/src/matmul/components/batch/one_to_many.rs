use std::marker::PhantomData;

use crate::matmul::components::batch::shared::gmm_execute;
use crate::matmul::components::{
    batch, config::MatmulConfig, global, Ident, MatmulKernel, MatmulLaunch, StageDim,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::Config as _;

/// Performs matrix multiplication at the batch level,
/// with one cube assigned to several underlying global matmuls
pub struct Matmul<
    EG: Numeric,
    ES: Numeric,
    GMM: global::Matmul<
        EG,
        ES,
        global::tensor_view::LhsLoader<EG, ES>,
        global::tensor_view::RhsLoader<EG, ES>,
        global::tensor_view::Unloader<EG>,
    >,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _gmm: PhantomData<GMM>,
}

#[cube]
impl<
        EG: Numeric,
        ES: Numeric,
        GMM: global::Matmul<
            EG,
            ES,
            global::tensor_view::LhsLoader<EG, ES>,
            global::tensor_view::RhsLoader<EG, ES>,
            global::tensor_view::Unloader<EG>,
        >,
    > batch::Matmul<EG> for Matmul<EG, ES, GMM>
{
    fn execute(
        lhs: Tensor<Line<EG>>,
        rhs: Tensor<Line<EG>>,
        out: Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    ) {
        let rank = out.rank();
        let shape_x = out.shape(rank - 2);
        let shape_y = out.shape(rank - 1);
        let mut shape_z = 1;
        #[unroll]
        for b in 0..rank - 2 {
            shape_z *= out.shape(b);
        }

        let cubes_x = config.cube_count_x();
        let cubes_y = config.cube_count_y();
        let cubes_z = config.cube_count_z();

        let stage_x = config.stage_dim(Ident::Out).num_elements_x_dim();
        let stage_y = config.stage_dim(Ident::Out).num_elements_y_dim();
        let stage_z = 1;

        let num_stages_x = (shape_x + stage_x - 1) / stage_x;
        let num_stages_y = (shape_y + stage_y - 1) / stage_y;
        let num_stages_z = (shape_z + stage_z - 1) / stage_z;

        // Each cube must do (span_x, span_y, span_z) times the matmul
        let span_x = num_stages_x / cubes_x;
        let span_y = num_stages_y / cubes_y;
        let span_z = num_stages_z / cubes_z;

        let cube_offset_x = CUBE_POS_X * span_x;
        let cube_offset_y = CUBE_POS_Y * span_y;
        let cube_offset_z = CUBE_POS_Z * span_z;

        let k_range = (0, lhs.shape(rank - 1));
        let gmm_config = config.to_gmm_config();

        // Outer is batch, as there's no hope of hitting L2 cache for batch
        for nth_batch in cube_offset_z..cube_offset_z + span_z {
            // Row/col/swizzle shall impact here. This is row major
            for x_offset in cube_offset_x..cube_offset_x + span_x {
                for y_offset in cube_offset_y..cube_offset_y + span_y {
                    gmm_execute::<EG, ES, GMM>(
                        lhs, rhs, out, x_offset, y_offset, nth_batch, k_range, gmm_config,
                    );
                }
            }
        }
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        GMM: global::Matmul<
            EG,
            ES,
            global::tensor_view::LhsLoader<EG, ES>,
            global::tensor_view::RhsLoader<EG, ES>,
            global::tensor_view::Unloader<EG>,
        >,
    > MatmulKernel<EG, EG> for Matmul<EG, ES, GMM>
{
    type Config = Config<GMM::Config>;

    fn check_config(config: Self::Config) {
        GMM::check_config(config.to_gmm_config())
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        GMM: global::Matmul<
            EG,
            ES,
            global::tensor_view::LhsLoader<EG, ES>,
            global::tensor_view::RhsLoader<EG, ES>,
            global::tensor_view::Unloader<EG>,
        >,
    > MatmulLaunch<EG, EG> for Matmul<EG, ES, GMM>
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
        self.cube_count_z
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
