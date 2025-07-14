use cubecl_core::CubeDim;

use crate::components::batch::{BatchConfig, HypercubeConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyBatchConfig {}

impl BatchConfig for DummyBatchConfig {
    fn hypercube_config(&self) -> HypercubeConfig {
        todo!()
    }

    fn cube_dim(&self) -> CubeDim {
        todo!()
    }
}
