use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::MaskTile;
use crate::components::tile::RowWise;
use cubecl_std::tensor::layout::Coords2d;

#[cube]
pub trait FragmentLayout: CubeType {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d;
    fn num_units_per_row(&self) -> comptime_type!(u32);
}

#[cube]
pub trait FragmentOps<E: Float> {
    type Layout: FragmentLayout;

    fn rowwise_max(&self) -> RowWise<E>;
    fn rowwise_sum(&self) -> RowWise<E>;

    fn scale(&mut self, val: &RowWise<E>);
    fn scale_and_mask<M: MaskTile>(this: &mut Self, scale: E, mask: &M);
    fn exp_m_diff(&mut self, m: &RowWise<E>);

    fn layout(&self) -> Self::Layout;
}

#[cube]
pub trait FragmentMask: CubeType {
    fn should_mask(&self, local_pos: Coords2d) -> bool;
}
