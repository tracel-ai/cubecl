use std::{fmt::Debug, hash::Hash};

use crate::components::{MatrixLayout, global::memory::ViewDirection};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalMemoryConfig {
    pub line_size: u32,
    pub check_row_bounds: bool,
    pub check_col_bounds: bool,
    pub matrix_layout: MatrixLayout,
    pub view_direction: ViewDirection,
}

impl GlobalMemoryConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        line_size: u32,
        check_row_bounds: bool,
        check_col_bounds: bool,
        matrix_layout: MatrixLayout,
        view_direction: ViewDirection,
    ) -> Self {
        GlobalMemoryConfig {
            line_size,
            check_row_bounds,
            check_col_bounds,
            matrix_layout,
            view_direction,
        }
    }
}
