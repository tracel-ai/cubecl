use cubecl_core::CubeDim;

use crate::components::{
    GlobalPartitionSize, MatmulLineSizes, MatmulProblem, MatmulSetupError,
    batch::{BatchConfig, HypercubeConfig},
    global::GlobalConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for partitioned batch matmul
pub struct PartitionedBatchConfig<G: GlobalConfig> {
    global_config: G,
    hypercube_config: HypercubeConfig,
    pub global_partition_size: GlobalPartitionSize,
}

impl<G: GlobalConfig> BatchConfig for PartitionedBatchConfig<G> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn cube_dim(&self) -> CubeDim {
        self.global_config.cube_dim()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        self.global_config.global_line_sizes()
    }

    fn hypercube_config(&self) -> HypercubeConfig {
        self.hypercube_config
    }

    fn can_yield_extra_cubes(&self) -> bool {
        self.hypercube_config
            .cube_count_plan_config
            .can_yield_extra_cubes()
    }
}

impl<G: GlobalConfig> PartitionedBatchConfig<G> {
    /// Create a new config for partitioned batch matmul
    pub fn new(
        global_config: G,
        hypercube_config: HypercubeConfig,
        global_partition_size: GlobalPartitionSize,
    ) -> Self {
        Self {
            global_config,
            hypercube_config,
            global_partition_size,
        }
    }

    /// May return an error if:
    /// - hypercube config is invalid
    pub fn validate(self, problem: &MatmulProblem) -> Result<Self, MatmulSetupError> {
        self.hypercube_config.validate(problem)?;
        Ok(self)
    }
}
