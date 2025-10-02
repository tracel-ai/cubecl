use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait PlaneLayout: CubeType {
    type E: Float;

    fn num_local_rows(&self) -> comptime_type!(u32);
    fn num_local_cols(&self) -> comptime_type!(u32);
    fn num_units_per_row(&self) -> comptime_type!(u32);

    fn get_at_coor(&self, local_row: u32, local_col: u32) -> Self::E;
    fn scale_at_coor(&mut self, local_row: u32, local_col: u32, val: Self::E);
    // fn scale_at_coor_tmp(&mut self, local_row: u32, local_col: u32, val: Self::E);
    fn exp_m_diff_at_coor(&mut self, local_row: u32, local_col: u32, m: Self::E);
}

#[cube]
pub trait RowWise: CubeType {
    type E: Float;

    fn new_filled(#[comptime] num_rows: u32, val: Self::E) -> Self;
    fn new_min_value(#[comptime] num_rows: u32) -> Self;
    fn new_zero(#[comptime] num_rows: u32) -> Self;

    fn copy_from(this: &mut Self, other: &Self);
    fn index(&self, i: u32) -> Self::E;

    fn row_sum<PL: PlaneLayout<E = Self::E>>(placeholder: &mut Self, data: &PL);
    fn row_max<PL: PlaneLayout<E = Self::E>>(placeholder: &mut Self, base: &Self, data: &PL);
}
