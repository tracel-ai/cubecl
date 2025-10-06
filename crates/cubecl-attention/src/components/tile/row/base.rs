use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::TileMask;
use crate::components::tile::RowWise;

#[cube]
pub trait PlaneLayout<E: Float>: CubeType {
    fn num_local_rows(&self) -> comptime_type!(u32);
    fn num_local_cols(&self) -> comptime_type!(u32);
    fn num_units_per_row(&self) -> comptime_type!(u32);

    fn rowwise_max(&self) -> RowWise<E>;
    fn rowwise_sum(&self) -> RowWise<E>;

    fn scale(&mut self, val: &RowWise<E>);
    fn scale_and_mask(&mut self, scale: E, mask: TileMask);
    fn exp_m_diff(&mut self, m: &RowWise<E>);
}
