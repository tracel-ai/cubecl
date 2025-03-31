use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::stage::{StageConfig, TilingLayout};
use crate::matmul::components::tile::Tile;
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{PhysicalDualStage, VirtualDualStage};

/// Determines which [DualStage] to use
#[derive(CubeType, Clone, Copy)]
pub enum DualStageFormat {
    Virtual,
    Physical,
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for double buffering global matmuls
pub enum DualStage<ES: Numeric, T: TilingLayout> {
    Virtual(VirtualDualStage<ES, T>),
    Physical(PhysicalDualStage<ES, T>),
}

#[cube]
impl<ES: Numeric, T: TilingLayout> DualStage<ES, T> {
    pub fn new<S: StageConfig>(
        dual_stage_format: DualStageFormat,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Self {
        match dual_stage_format {
            DualStageFormat::Virtual => {
                DualStage::new_Virtual(VirtualDualStage::new::<S>(ident, config))
            }
            DualStageFormat::Physical => {
                DualStage::new_Physical(PhysicalDualStage::new::<S>(ident, config))
            }
        }
    }

    pub fn clear_buffer<S: StageConfig>(
        &mut self,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) {
        match self {
            DualStage::Virtual(virtual_dual_stage) => {
                virtual_dual_stage.clear_buffer::<S>(buffer_id, ident, config)
            }
            DualStage::Physical(physical_dual_stage) => {
                physical_dual_stage.clear_buffer::<S>(buffer_id, ident, config)
            }
        }
    }

    /// Get the tile at position (x,y) regardless of matrix layout
    pub fn get_tile<S: StageConfig>(
        &self,
        x_in_buffer: u32,
        y_in_buffer: u32,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        match self {
            DualStage::Virtual(virtual_dual_stage) => {
                virtual_dual_stage.get_tile::<S>(x_in_buffer, y_in_buffer, buffer_id, ident, config)
            }
            DualStage::Physical(physical_dual_stage) => physical_dual_stage.get_tile::<S>(
                x_in_buffer,
                y_in_buffer,
                buffer_id,
                ident,
                config,
            ),
        }
    }
}
