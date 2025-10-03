use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::TileMask;
use crate::components::tile::dummy::AttentionMatmulConfig;

#[cube]
pub trait PlaneLayout: CubeType {
    type E: Float;

    fn num_local_rows(&self) -> comptime_type!(u32);
    fn num_local_cols(&self) -> comptime_type!(u32);
    fn num_units_per_row(&self) -> comptime_type!(u32);

    fn get_at_coor(&self, local_row: u32, local_col: u32) -> Self::E;
    fn scale(&mut self, val: Self::E);
    fn scale_and_mask(&mut self, scale: Self::E, mask: TileMask);
    fn exp_m_diff(&mut self, m: Self::E);
}

#[cube]
pub trait RowWise: CubeType {
    type E: Float;

    fn new_filled(#[comptime] num_rows: u32, val: Self::E) -> Self;
    fn new_min_value(#[comptime] num_rows: u32) -> Self;
    fn new_zero(#[comptime] num_rows: u32) -> Self;

    fn copy_from(this: &mut Self, other: &Self);
    fn index(&self, i: u32) -> Self::E;
    fn fill(this: &mut Self, val: Self::E);

    fn row_sum<PL: PlaneLayout<E = Self::E>, TC: AttentionMatmulConfig>(
        placeholder: &mut Self,
        data: &PL,
        #[comptime] config: TC,
    );
    fn row_max<PL: PlaneLayout<E = Self::E>, TC: AttentionMatmulConfig>(
        placeholder: &mut Self,
        base: &Self,
        data: &PL,
        #[comptime] config: TC,
    );
}
