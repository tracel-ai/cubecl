use std::{fmt::Debug, hash::Hash};

use crate::components::MatrixLayout;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct StageMemoryConfig {
    // Planes that read or write this stage memory
    pub num_planes: u32,
    pub elements_per_tile_along_row: u32,
    pub elements_per_tile_along_col: u32,
    pub tiles_per_partition_along_row: u32,
    pub tiles_per_partition_along_col: u32,
    pub partitions_per_stage_along_row: u32,
    pub partitions_per_stage_along_col: u32,
    pub line_size: u32,
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

impl SwizzleMode {
    pub fn atom_size(&self) -> usize {
        match self {
            SwizzleMode::None => usize::MAX,
            SwizzleMode::B32 | SwizzleMode::B64 | SwizzleMode::B128 => 16,
        }
    }

    pub fn span_size(&self) -> usize {
        match self {
            SwizzleMode::None => 1,
            SwizzleMode::B32 => 32,
            SwizzleMode::B64 => 64,
            SwizzleMode::B128 => 128,
        }
    }
}

impl StageMemoryConfig {
    pub fn tiles_per_stage_along_row(&self) -> u32 {
        self.tiles_per_partition_along_row * self.partitions_per_stage_along_row
    }

    pub fn tiles_per_stage_along_col(&self) -> u32 {
        self.tiles_per_partition_along_col * self.partitions_per_stage_along_col
    }

    pub fn elements_per_stage_along_row(&self) -> u32 {
        self.tiles_per_stage_along_row() * self.elements_per_tile_along_row
    }

    pub fn elements_per_stage_along_col(&self) -> u32 {
        self.tiles_per_stage_along_col() * self.elements_per_tile_along_col
    }

    pub fn elements_per_tile(&self) -> u32 {
        self.elements_per_tile_along_row * self.elements_per_tile_along_col
    }

    pub fn elements_per_stage(&self) -> u32 {
        self.elements_per_stage_along_row() * self.elements_per_stage_along_col()
    }

    pub fn tiles_per_stage(&self) -> u32 {
        self.tiles_per_stage_along_row() * self.tiles_per_stage_along_col()
    }

    pub fn elements_per_tile_along_contiguous_dim(&self) -> u32 {
        match self.matrix_layout {
            MatrixLayout::RowMajor => self.elements_per_tile_along_col,
            MatrixLayout::ColMajor => self.elements_per_tile_along_row,
        }
    }

    pub fn elements_per_stage_along_contiguous_dim(&self) -> u32 {
        match self.matrix_layout {
            MatrixLayout::RowMajor => self.elements_per_stage_along_col(),
            MatrixLayout::ColMajor => self.elements_per_stage_along_row(),
        }
    }
}
