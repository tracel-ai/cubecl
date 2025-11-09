use std::{fmt::Debug, hash::Hash};

use crate::components::MatrixLayout;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct StageMemoryConfig {
    pub num_main_flow_planes: u32,
    pub elements_in_tile_row: u32,
    pub elements_in_tile_col: u32,
    pub tiles_in_stage_row: u32,
    pub tiles_in_stage_col: u32,
    pub stage_line_size: u32,
    pub matrix_layout: MatrixLayout,
    pub swizzle: SwizzleMode,
    pub num_stages: u32,
}

/// Swizzling mode of the shared memory. Default `None`.
/// Matches the base TMA functionality, alternative chunk sizes or more complex patterns don't really
/// apply to matmul.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum SwizzleMode {
    /// No swizzling
    #[default]
    None,
    /// Swizzle 16B chunks within 32B span
    /// Swizzle<1,4,3>
    B32,
    /// Swizzle 16B chunks within 64B span
    /// Swizzle<2,4,3>
    B64,
    /// Swizzle 16B chunks within 128B span
    /// Swizzle<3,4,3>
    B128,
}

impl StageMemoryConfig {
    pub fn elements_in_stage_row(&self) -> u32 {
        self.tiles_in_stage_row * self.elements_in_tile_row
    }

    pub fn elements_in_stage_col(&self) -> u32 {
        self.tiles_in_stage_col * self.elements_in_tile_col
    }

    pub fn elements_in_stage(&self) -> u32 {
        self.elements_in_stage_row() * self.elements_in_stage_col()
    }

    pub fn elements_in_tile(&self) -> u32 {
        self.elements_in_tile_row * self.elements_in_tile_col
    }
}
