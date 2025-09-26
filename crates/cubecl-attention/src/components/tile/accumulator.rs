use crate::components::tile::RowWise;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait AccumulatorTile<E: Float>: CubeType {
    type Fragment: AccumulatorFragment<E>;

    fn zero(&mut self);
    // tmp
    fn scale(&mut self, scale: &RowWise<E>, #[comptime] scale_op: ScaleMode);
}

#[cube]
pub trait AccumulatorFragment<E: Float>: CubeType {}

pub enum ScaleMode {
    Multiply,
    Divide,
}
