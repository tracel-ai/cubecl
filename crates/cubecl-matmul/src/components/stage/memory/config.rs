use std::{fmt::Debug, hash::Hash};

use crate::components::MatrixLayout;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct StageMemoryConfig {
    // Planes that read this stage memory
    pub num_reading_planes: u32,
    pub elements_in_tile_row: u32,
    pub elements_in_tile_col: u32,
    pub tiles_in_stage_row: u32,
    pub tiles_in_stage_col: u32,
    pub line_size: u32,
    pub matrix_layout: MatrixLayout,
    pub num_stages: u32,
}

impl StageMemoryConfig {
    pub fn elements_in_stage_row(&self) -> u32 {
        self.tiles_in_stage_row * self.elements_in_tile_row
    }

    pub fn elements_in_stage_col(&self) -> u32 {
        self.tiles_in_stage_col * self.elements_in_tile_col
    }

    pub fn elements_in_tile(&self) -> u32 {
        self.elements_in_tile_row * self.elements_in_tile_col
    }

    pub fn elements_in_stage(&self) -> u32 {
        self.elements_in_stage_row() * self.elements_in_stage_col()
    }

    pub fn tiles_in_stage(&self) -> u32 {
        self.tiles_in_stage_row * self.tiles_in_stage_col
    }

    pub fn elements_in_tile_contiguous_dim(&self) -> u32 {
        match self.matrix_layout {
            MatrixLayout::RowMajor => self.elements_in_tile_col,
            MatrixLayout::ColMajor => self.elements_in_tile_row,
        }
    }

    pub(crate) fn elements_in_stage_contiguous_dim(&self) -> u32 {
        match self.matrix_layout {
            MatrixLayout::RowMajor => self.elements_in_stage_col(),
            MatrixLayout::ColMajor => self.elements_in_stage_row(),
        }
    }
}
