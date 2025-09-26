use crate::components::tile::RowWise;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait AccumulatorTile<E: Float>: CubeType {
    fn scale(&mut self, scale: &RowWise<E>, #[comptime] scale_op: ScaleMode);
}

pub enum ScaleMode {
    Multiply,
    Divide,
}
