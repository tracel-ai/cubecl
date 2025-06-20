use cubecl_core::{CubeCount, CubeDim};

use crate::{
    components::{
        Ident, MatmulConfig, MatmulLineSizes, MatmulProblem,
        batch::{BatchConfig, CubeCounterConfig, GlobalPartitioning},
        global::GlobalConfig,
    },
    kernels::MatmulSetupError,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartitionedBatchConfig<G: GlobalConfig> {
    global_config: G,
    num_sms: u32,
    global_partitioning: GlobalPartitioning,
}

impl<G: GlobalConfig> BatchConfig for PartitionedBatchConfig<G> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn quantized(&self) -> bool {
        self.global_config().quantized()
    }

    fn cube_dim(&self) -> CubeDim {
        self.global_config.cube_dim()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        MatmulLineSizes {
            lhs: self.global_config.global_line_size(Ident::Lhs) as u8,
            rhs: self.global_config.global_line_size(Ident::Rhs) as u8,
            out: self.global_config.global_line_size(Ident::Out) as u8,
        }
    }

    fn cube_count(&self, problem: &MatmulProblem) -> CubeCount {
        self.cube_counter_config().cube_count(problem)
    }

    fn cube_counter_config(&self) -> CubeCounterConfig {
        CubeCounterConfig::new(
            self.num_sms,
            &self.tiling_scheme(),
            self.global_partitioning,
        )
    }
}

impl<G: GlobalConfig> MatmulConfig for PartitionedBatchConfig<G> {}

impl<G: GlobalConfig> PartitionedBatchConfig<G> {
    pub fn new(
        global_config: G,
        num_sms: u32,
        global_partitioning: GlobalPartitioning,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            global_config,
            num_sms,
            global_partitioning,
        }
        .validate()
    }

    fn validate(self) -> Result<Self, MatmulSetupError> {
        Ok(self)
    }
}
