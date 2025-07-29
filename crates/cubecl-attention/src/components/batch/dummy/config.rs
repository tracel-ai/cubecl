use cubecl_core::CubeDim;

use crate::components::{
    AttentionProblem, AttentionSetupError,
    batch::{BatchConfig, HypercubeConfig},
    global::GlobalConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyBatchConfig<G: GlobalConfig> {
    global_config: G,
    hypercube_config: HypercubeConfig,
    seq_k: u32,
}

impl<G: GlobalConfig> BatchConfig for DummyBatchConfig<G> {
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

    fn seq_k(&self) -> u32 {
        self.seq_k
    }
}

impl<G: GlobalConfig> DummyBatchConfig<G> {
    pub fn new(global_config: G, hypercube_config: HypercubeConfig, seq_k: u32) -> Self {
        Self {
            global_config,
            hypercube_config,
            seq_k,
        }
    }

    pub fn validate(self, _problem: &AttentionProblem) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
