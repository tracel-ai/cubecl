use std::{fmt::Debug, hash::Hash};

use crate::components::MatrixLayout;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct GlobalMemoryConfig {
    elements_in_tile_row: u32,
    elements_in_tile_col: u32,
    elements_in_stage_row: u32,
    elements_in_stage_col: u32,
    global_line_size: u32,
    check_row_bounds: bool,
    check_col_bounds: bool,
    matrix_layout: MatrixLayout,
}

impl GlobalMemoryConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        elements_in_tile_row: u32,
        elements_in_tile_col: u32,
        elements_in_stage_row: u32,
        elements_in_stage_col: u32,
        global_line_size: u32,
        check_row_bounds: bool,
        check_col_bounds: bool,
        matrix_layout: MatrixLayout,
    ) -> Self {
        GlobalMemoryConfig {
            elements_in_tile_row,
            elements_in_tile_col,
            elements_in_stage_row,
            elements_in_stage_col,
            global_line_size,
            check_row_bounds,
            check_col_bounds,
            matrix_layout,
        }
    }

    pub fn matrix_layout(&self) -> MatrixLayout {
        self.matrix_layout
    }

    pub fn elements_in_tile(&self) -> u32 {
        self.elements_in_tile_row() * self.elements_in_tile_col()
    }

    pub fn elements_in_tile_row(&self) -> u32 {
        self.elements_in_tile_row
    }

    pub fn elements_in_tile_col(&self) -> u32 {
        self.elements_in_tile_col
    }

    pub fn elements_in_stage(&self) -> u32 {
        self.elements_in_stage_row() * self.elements_in_stage_col()
    }

    pub fn elements_in_stage_row(&self) -> u32 {
        self.elements_in_stage_row
    }

    pub fn elements_in_stage_col(&self) -> u32 {
        self.elements_in_stage_col
    }

    pub fn line_size(&self) -> u32 {
        self.global_line_size
    }

    pub fn check_row_bounds(&self) -> bool {
        self.check_row_bounds
    }

    pub fn check_col_bounds(&self) -> bool {
        self.check_col_bounds
    }
}
