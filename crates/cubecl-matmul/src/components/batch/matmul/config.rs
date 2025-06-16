use cubecl_core::{CubeDim, Runtime, client::ComputeClient};

use crate::{
    components::{
        Ident, MatmulConfig, MatmulLineSizes, MatmulPrecision, batch::BatchConfig,
        global::GlobalConfig,
    },
    kernels::MatmulSetupError,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartitionedBatchConfig<G: GlobalConfig> {
    global_config: G,
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
}

impl<G: GlobalConfig> MatmulConfig for PartitionedBatchConfig<G> {}

impl<G: GlobalConfig> PartitionedBatchConfig<G> {
    pub fn new<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        global_config: G,
    ) -> Result<Self, MatmulSetupError> {
        Self { global_config }.validate()
    }

    fn validate(self) -> Result<Self, MatmulSetupError> {
        Ok(self)
    }
}
