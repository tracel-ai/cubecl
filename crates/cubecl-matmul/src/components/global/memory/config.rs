use std::{fmt::Debug, hash::Hash};

use crate::components::MatrixLayout;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalMemoryConfig {
    pub elements_in_tile_row: u32,
    pub elements_in_tile_col: u32,
    pub elements_in_stage_row: u32,
    pub elements_in_stage_col: u32,
    pub global_line_size: u32,
    pub check_row_bounds: bool,
    pub check_col_bounds: bool,
    pub matrix_layout: MatrixLayout,
}
