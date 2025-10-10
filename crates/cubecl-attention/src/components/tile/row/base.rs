use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::RowWise;
use cubecl_std::tensor::layout::Coords2d;

#[cube]
pub trait FragmentLayout: CubeType {
    fn num_local_rows(&self) -> comptime_type!(u32);
    fn num_local_cols(&self) -> comptime_type!(u32);
    fn num_units_per_row(&self) -> comptime_type!(u32);
}

#[cube]
pub trait FragmentOps<E: Float>: FragmentLayout {
    fn rowwise_max(&self) -> RowWise<E>;
    fn rowwise_sum(&self) -> RowWise<E>;

    fn scale(&mut self, val: &RowWise<E>);
    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M);
    fn exp_m_diff(&mut self, m: &RowWise<E>);
}

#[cube]
pub trait FragmentMask: FragmentLayout {
    fn apply<E: Float>(this: &Self, pos: Coords2d) -> E;
}
