use cubecl_core::CubeDim;

use crate::components::{
    batch::{BatchAttentionConfig, HypercubeConfig},
    global::GlobalAttentionConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SimpleBatchConfig<G: GlobalAttentionConfig> {
    global_config: G,
    hypercube_config: HypercubeConfig,
}

impl<G: GlobalAttentionConfig> BatchAttentionConfig for SimpleBatchConfig<G> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn hypercube_config(&self) -> HypercubeConfig {
        self.hypercube_config
    }

    fn cube_dim(&self) -> CubeDim {
        self.global_config.cube_dim()
    }
}

impl<G: GlobalAttentionConfig> SimpleBatchConfig<G> {
    pub fn new(global_config: G, hypercube_config: HypercubeConfig) -> Self {
        Self {
            global_config,
            hypercube_config,
        }
    }
}
