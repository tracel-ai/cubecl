use cubecl_core::{CubeCount, CubeDim};

use crate::{
    components::{
        Ident, MatmulConfig, MatmulLineSizes, MatmulProblem,
        batch::{BatchConfig, CubeCounterConfig},
        global::GlobalConfig,
    },
    kernels::MatmulSetupError,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartitionedBatchConfig<G: GlobalConfig> {
    global_config: G,
    cube_counter_config: CubeCounterConfig,
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
        let x = self
            .cube_counter_config
            .cube_count_data(problem)
            .to_cube_count();
        println!("cube_count{:?}", x);
        x
    }

    fn cube_counter_config(&self) -> CubeCounterConfig {
        self.cube_counter_config
    }
}

impl<G: GlobalConfig> MatmulConfig for PartitionedBatchConfig<G> {}

impl<G: GlobalConfig> PartitionedBatchConfig<G> {
    pub fn new(
        global_config: G,
        cube_counter_config: CubeCounterConfig,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            global_config,
            cube_counter_config,
        }
        .validate()
    }

    fn validate(self) -> Result<Self, MatmulSetupError> {
        Ok(self)
    }
}
