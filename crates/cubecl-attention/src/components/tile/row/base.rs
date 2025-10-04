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
    fn scale_and_mask(&mut self, scale: &RowWise<E>, mask: TileMask);
    fn exp_m_diff(&mut self, m: &RowWise<E>);
}

// #[cube]
// pub trait RowWise: CubeType {
//     type E: Float;

//     fn new_filled(#[comptime] num_rows: u32, val: Self::E) -> Self;
//     fn new_min_value(#[comptime] num_rows: u32) -> Self;
//     fn new_zero(#[comptime] num_rows: u32) -> Self;

//     fn copy_from(this: &mut Self, other: &Self);
//     fn index(&self, i: u32) -> Self::E;
//     fn fill(this: &mut Self, val: Self::E);

//     fn row_sum<PL: PlaneLayout<RW = Self>, TC: AttentionMatmulConfig>(
//         placeholder: &mut Self,
//         data: &PL,
//         #[comptime] config: TC,
//     );
//     fn row_max<PL: PlaneLayout<RW = Self>, TC: AttentionMatmulConfig>(
//         placeholder: &mut Self,
//         base: &Self,
//         data: &PL,
//         #[comptime] config: TC,
//     );
// }
