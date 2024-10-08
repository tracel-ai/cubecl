use cubecl_core::{
    client::ComputeClient,
    ir::{Elem, FloatKind},
    Feature, Runtime,
};

use crate::matmul::cmma::config::CmmaConfig;

use super::config::TileDimension;

#[derive(Debug)]
pub enum UnavailabilityReason {
    HighlyPermutatedInput,
    SharedMemoryLimitBusted,
    InvalidConfig(String),
    CmmaInstructionsUnsupported,
}

/// Checks if the matmul cmma can be used.
pub fn check_cmma_availability<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    cmma_config: &CmmaConfig,
) -> Result<(), UnavailabilityReason> {
    let tile_dim: TileDimension = cmma_config.tile_dimension_strategy.into();
    if !client.properties().feature_enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: tile_dim.m as u8,
        k: tile_dim.k as u8,
        n: tile_dim.n as u8,
    }) {
        return Err(UnavailabilityReason::CmmaInstructionsUnsupported);
    }

    Ok(())
}
