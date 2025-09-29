use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait RowFormat: CubeType {
    type RowElement<E: Float>: RowElement<E>;

    fn new_filled<E: Float>(value: E) -> Self::RowElement<E>;
}

#[cube]
pub trait RowElement<E: Float>: CubeType {
    fn copy(from: &Self, to: &mut Self);
}
