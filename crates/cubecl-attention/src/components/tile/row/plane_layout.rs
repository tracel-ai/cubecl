use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait PlaneLayout: CubeType {
    /// Number of logical rows this thread handles
    fn num_rows(&self) -> comptime_type!(u32);

    /// Maps `r ∈ [0..num_rows)` to the absolute row index
    fn row_index(&self, r: u32) -> u32;

    /// Number of columns in logical row `r`
    fn num_cols(&self, r: u32) -> u32;

    /// Maps `(r, c)` with `c ∈ [0..num_cols(r))` to the absolute column index
    fn col_index(&self, r: u32, c: u32) -> u32;
}
