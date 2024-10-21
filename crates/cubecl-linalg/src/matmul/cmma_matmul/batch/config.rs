use crate::matmul::{
    config::{ComptimeConfig, MatmulConfig, MatmulLaunchConfig},
    matmul_batch::BmmConfig,
    matmul_global::GmmConfig,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaBatchMatmulConfig<G: GmmConfig> {
    gmm_config: G,
}

impl<G: GmmConfig> BmmConfig for CmmaBatchMatmulConfig<G> {
    type GmmConfig = G;

    fn to_gmm_config(&self) -> Self::GmmConfig {
        self.gmm_config
    }
}

impl<G: GmmConfig> ComptimeConfig for CmmaBatchMatmulConfig<G> {}
impl<G: GmmConfig> MatmulConfig for CmmaBatchMatmulConfig<G> {}

impl<G: GmmConfig> MatmulLaunchConfig for CmmaBatchMatmulConfig<G> {
    fn cube_dim(&self) -> CubeDim {
        CubeDim {
            x: self.gmm_config.plane_dim(),
            y: self.gmm_config.num_planes(),
            z: 1,
        }
    }

    fn cube_count(&self) -> CubeCount {
        CubeCount::Static(1, 1, 1)
    }
}

impl<G: GmmConfig> CmmaBatchMatmulConfig<G> {
    pub fn new(gmm_config: G) -> Self {
        Self { gmm_config }
    }
}
