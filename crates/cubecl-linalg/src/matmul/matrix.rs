use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ident {
    Lhs,
    Rhs,
    Out,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}

#[cube]
pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
    match layout {
        MatrixLayout::RowMajor => cmma::MatrixLayout::RowMajor,
        MatrixLayout::ColMajor => cmma::MatrixLayout::ColMajor,
    }
}
