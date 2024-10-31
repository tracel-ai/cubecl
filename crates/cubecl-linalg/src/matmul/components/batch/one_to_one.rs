use std::marker::PhantomData;

use crate::matmul::components::batch::BatchMatmul;
use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::global::{
    GlobalMatmul, GmmConfig, LhsTensorLoader, RhsTensorLoader, TensorUnloader,
};
use crate::matmul::components::matrix::Ident;
use crate::matmul::components::stage_dim::StageDim;
use crate::matmul::components::{MatmulKernel, MatmulLaunch};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::BmmConfig;

/// Performs matrix multiplication at the batch level,
/// with one cube assigned to each underlying global matmul
pub struct Matmul<
    EG: Numeric,
    ES: Numeric,
    GMM: GlobalMatmul<
        EG,
        ES,
        LhsTensorLoader<EG, ES, B::GmmConfig>,
        RhsTensorLoader<EG, ES, B::GmmConfig>,
        TensorUnloader<EG, B::GmmConfig>,
        B::GmmConfig,
    >,
    B: BmmConfig,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _gmm: PhantomData<GMM>,
    _config: PhantomData<B>,
}

#[cube]
impl<
        EG: Numeric,
        ES: Numeric,
        GMM: GlobalMatmul<
            EG,
            ES,
            LhsTensorLoader<EG, ES, B::GmmConfig>,
            RhsTensorLoader<EG, ES, B::GmmConfig>,
            TensorUnloader<EG, B::GmmConfig>,
            B::GmmConfig,
        >,
        B: BmmConfig,
    > BatchMatmul<EG, B> for Matmul<EG, ES, GMM, B>
{
    fn execute(
        lhs: Tensor<Line<EG>>,
        rhs: Tensor<Line<EG>>,
        out: Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    ) {
        // TODO row/col/swizzle
        let x_offset = CUBE_POS_X * config.stage_dim(Ident::Lhs).num_elements_x_dim();
        let y_offset = CUBE_POS_Y * config.stage_dim(Ident::Rhs).num_elements_y_dim();
        let nth_batch = CUBE_POS_Z;
        let k_range = (0, lhs.shape(lhs.rank() - 1));

        GMM::execute(
            LhsTensorLoader::new(lhs, x_offset, k_range.0, nth_batch, config.to_gmm_config()),
            RhsTensorLoader::new(rhs, k_range.0, y_offset, nth_batch, config.to_gmm_config()),
            TensorUnloader::new(out, x_offset, y_offset, nth_batch),
            k_range,
            config.to_gmm_config(),
        );
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        GMM: GlobalMatmul<
            EG,
            ES,
            LhsTensorLoader<EG, ES, B::GmmConfig>,
            RhsTensorLoader<EG, ES, B::GmmConfig>,
            TensorUnloader<EG, B::GmmConfig>,
            B::GmmConfig,
        >,
        B: BmmConfig,
    > MatmulKernel<EG, EG> for Matmul<EG, ES, GMM, B>
{
    type Config = B;

    fn check_config(config: Self::Config) {
        GMM::check_config(config.to_gmm_config())
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        GMM: GlobalMatmul<
            EG,
            ES,
            LhsTensorLoader<EG, ES, B::GmmConfig>,
            RhsTensorLoader<EG, ES, B::GmmConfig>,
            TensorUnloader<EG, B::GmmConfig>,
            B::GmmConfig,
        >,
        B: BmmConfig,
    > MatmulLaunch<EG, EG> for Matmul<EG, ES, GMM, B>
{
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: B,
    ) {
        Self::check_config(config);
        batch_matmul_launch::launch_unchecked::<EG, ES, Self, Self::Config, R>(
            &client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}

#[cube(launch_unchecked)]
// TODO input as references
pub(crate) fn batch_matmul_launch<
    EG: Numeric,
    ES: Numeric,
    BMM: BatchMatmul<EG, B>,
    B: BmmConfig,
>(
    lhs: Tensor<Line<EG>>,
    rhs: Tensor<Line<EG>>,
    out: Tensor<Line<EG>>,
    #[comptime] config: B,
) {
    BMM::execute(lhs, rhs, out, config);
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct Config<G: GmmConfig> {
    gmm_config: G,
    cube_count_x: u32,
    cube_count_y: u32,
    cube_count_z: u32,
}

impl<G: GmmConfig> BmmConfig for Config<G> {
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

impl<G: GmmConfig> MatmulConfig for Config<G> {}

impl<G: GmmConfig> Config<G> {
    pub fn new(gmm_config: G, cube_count_x: u32, cube_count_y: u32, cube_count_z: u32) -> Self {
        Self {
            gmm_config,
            cube_count_x,
            cube_count_y,
            cube_count_z,
        }
    }
}
