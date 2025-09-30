use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait PlaneLayout<E: Float>: CubeType {
    /// Number of logical rows this thread handles
    fn owned_rows_count(&self) -> comptime_type!(u32);

    /// Absolute number of rows in the tile
    fn total_rows_count(&self) -> comptime_type!(u32);

    /// Number of columns for a unit in one row
    fn num_cols(&self) -> comptime_type!(u32);

    /// Whether absolute number of row is part of owned rows
    fn is_owned(&self, row: u32) -> bool;

    /// Maps `r ∈ [0..num_rows)` to the absolute row index
    fn row_index(&self, r: u32) -> u32;

    /// Maps `(r, c)` with `c ∈ [0..num_cols(r))` to the absolute column index
    fn col_index(&self, r: u32, c: u32) -> u32;

    /// row and col are absolute (i.e. must get row_index, col_index beforehand)
    fn get_at_coor(&self, row: u32, col: u32, mask: E) -> E;
}
