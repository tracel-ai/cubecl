use crate::matmul::{
    config::{ComptimeConfig, MatmulConfig},
    matmul_batch::BmmConfig,
    matmul_global::GmmConfig,
    matrix::Ident,
    stage_dim::StageDim,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaBatchMatmulConfig<G: GmmConfig> {
    gmm_config: G,
    cube_count_x: u32,
    cube_count_y: u32,
    cube_count_z: u32,
}

impl<G: GmmConfig> BmmConfig for CmmaBatchMatmulConfig<G> {
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

impl<G: GmmConfig> ComptimeConfig for CmmaBatchMatmulConfig<G> {}
impl<G: GmmConfig> MatmulConfig for CmmaBatchMatmulConfig<G> {}

impl<G: GmmConfig> CmmaBatchMatmulConfig<G> {
    pub fn new(gmm_config: G, cube_count_x: u32, cube_count_y: u32, cube_count_z: u32) -> Self {
        Self {
            gmm_config,
            cube_count_x,
            cube_count_y,
            cube_count_z,
        }
    }
}
