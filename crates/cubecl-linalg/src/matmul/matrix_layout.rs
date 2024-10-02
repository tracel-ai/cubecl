use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}

impl CubeType for MatrixLayout {
    type ExpandType = Self;
}

impl Init for MatrixLayout {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl IntoRuntime for MatrixLayout {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        self
    }
}

#[cube]
pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
    match layout {
        MatrixLayout::RowMajor => cmma::MatrixLayout::RowMajor,
        MatrixLayout::ColMajor => cmma::MatrixLayout::ColMajor,
    }
}
