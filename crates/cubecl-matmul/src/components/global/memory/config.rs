use std::{fmt::Debug, hash::Hash};

use crate::components::{MatrixLayout, global::memory::ViewDirection};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalMemoryReadConfig {
    // pub elements_in_tile_row: u32,
    // pub elements_in_tile_col: u32,
    // pub elements_in_stage_row: u32,
    // pub elements_in_stage_col: u32,
    pub line_size: u32,
    pub check_row_bounds: bool,
    pub check_col_bounds: bool,
    pub matrix_layout: MatrixLayout,
    pub view_direction: ViewDirection,
}

impl GlobalMemoryReadConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // elements_in_tile_row: u32,
        // elements_in_tile_col: u32,
        // elements_in_stage_row: u32,
        // elements_in_stage_col: u32,
        line_size: u32,
        check_row_bounds: bool,
        check_col_bounds: bool,
        matrix_layout: MatrixLayout,
        view_direction: ViewDirection,
    ) -> Self {
        GlobalMemoryReadConfig {
            // elements_in_tile_row,
            // elements_in_tile_col,
            // elements_in_stage_row,
            // elements_in_stage_col,
            line_size,
            check_row_bounds,
            check_col_bounds,
            matrix_layout,
            view_direction,
        }
    }

    // pub fn matrix_layout(&self) -> MatrixLayout {
    //     self.matrix_layout
    // }

    // pub fn elements_in_tile(&self) -> u32 {
    //     self.elements_in_tile_row() * self.elements_in_tile_col()
    // }

    // pub fn elements_in_tile_row(&self) -> u32 {
    //     self.elements_in_tile_row
    // }

    // pub fn elements_in_tile_col(&self) -> u32 {
    //     self.elements_in_tile_col
    // }

    // pub fn elements_in_stage(&self) -> u32 {
    //     self.elements_in_stage_row() * self.elements_in_stage_col()
    // }

    // pub fn elements_in_stage_row(&self) -> u32 {
    //     self.elements_in_stage_row
    // }

    // pub fn elements_in_stage_col(&self) -> u32 {
    //     self.elements_in_stage_col
    // }

    //     pub fn line_size(&self) -> u32 {
    //         self.line_size
    //     }

    //     pub fn check_row_bounds(&self) -> bool {
    //         self.check_row_bounds
    //     }

    //     pub fn check_col_bounds(&self) -> bool {
    //         self.check_col_bounds
    //     }
}
