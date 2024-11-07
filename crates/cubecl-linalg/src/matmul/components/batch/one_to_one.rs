use std::marker::PhantomData;

use crate::matmul::components::{
    batch, config::MatmulConfig, global, Ident, MatmulKernel, MatmulLaunch, StageDim,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::Config as _;

/// Performs matrix multiplication at the batch level,
/// with one cube assigned to each underlying global matmul
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
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    ) {
        // TODO row/col/swizzle
        let x_offset = CUBE_POS_X * config.stage_dim(Ident::Lhs).num_elements_x_dim();
        let y_offset = CUBE_POS_Y * config.stage_dim(Ident::Rhs).num_elements_y_dim();
        let nth_batch = CUBE_POS_Z;
        let k_range = (0, lhs.shape(lhs.rank() - 1));

        GMM::execute(
            global::tensor_view::LhsLoader::new::<<Self::Config as batch::Config>::GmmConfig>(
                lhs,
                x_offset,
                k_range.0,
                nth_batch,
                config.to_gmm_config(),
            ),
            global::tensor_view::RhsLoader::new::<<Self::Config as batch::Config>::GmmConfig>(
                rhs,
                k_range.0,
                y_offset,
                nth_batch,
                config.to_gmm_config(),
            ),
            global::tensor_view::Unloader::new(out, x_offset, y_offset, nth_batch),
            k_range,
            config.to_gmm_config(),
        );
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
        launch::launch_unchecked::<EG, Self, R>(
            client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}

#[cube(launch_unchecked)]
fn launch<EG: Numeric, BMM: batch::Matmul<EG>>(
    lhs: &Tensor<Line<EG>>,
    rhs: &Tensor<Line<EG>>,
    out: &mut Tensor<Line<EG>>,
    #[comptime] config: BMM::Config,
) {
    BMM::execute(lhs, rhs, out, config);
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
        self.cube_count_x() * self.stage_dim(Ident::Out).num_elements_x_dim()
    }

    fn max_n(&self) -> u32 {
        self.cube_count_y() * self.stage_dim(Ident::Out).num_elements_y_dim()
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
}
