use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::RowWise;
use cubecl_std::tensor::layout::Coords2d;

#[cube]
/// Describes how a fragment is fragmented across units
/// The layout is independant of the data and data types
pub trait FragmentLayout: CubeType {
    /// Maps the (row, col) of the registers of a single unit to the position within the whole tile
    ///
    /// Example: for simplicity, if we had a 4 units warp for a 4x4 tile divided as such:
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    /// Then we would have:
    /// unit_0: absolute_pos((0, 0)) == (0, 0)
    /// unit_0: absolute_pos((0, 1)) == (0, 1)
    /// unit_0: absolute_pos((1, 0)) == (2, 0)
    /// unit_0: absolute_pos((1, 1)) == (2, 1)
    /// ...
    /// unit_3: absolute_pos((0, 0)) == (1, 2)
    /// unit_3: absolute_pos((0, 1)) == (1, 3)
    /// unit_3: absolute_pos((1, 0)) == (3, 2)
    /// unit_3: absolute_pos((1, 1)) == (3, 3)
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d;

    /// Gives how many units participate in the same row
    ///
    /// Example: for simplicity, if we had a 4 units warp for a 4x4 tile divided as such:
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    /// Then it would output 2, because each row is spread across two different units (0 and 1, or 2 and 3)
    /// Layouts with varying num_units_per_row are not supported
    fn num_units_per_row(&self) -> comptime_type!(u32);
}

#[cube]
/// Operations on a fragment, having a specific fragment layout
pub trait FragmentOps<E: Float> {
    /// How the fragment is fragmented across units
    type Layout: FragmentLayout;

    /// Get the layout of the fragment
    fn layout(&self) -> Self::Layout;

    /// Return the maximum of each row
    /// Units only output values for rows they participate in
    fn rowwise_max(&self) -> RowWise<E>;

    /// Return the sum of each row
    /// Units only output values for rows they participate in
    fn rowwise_sum(&self) -> RowWise<E>;

    /// Scale each element in a row by a value for this row
    fn rowwise_scale(&mut self, val: &RowWise<E>);

    /// Scale every element by a constant factor, and masks values identified by the mask
    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M);

    /// Changes each value x_ij for e^(x_ij - m_i) for every row
    fn exp_diff(&mut self, m: &RowWise<E>);
}

#[cube]
/// Describes which elements of a fragment should be masked
pub trait FragmentMask: CubeType {
    /// Returns `true` if the element at `local_pos` should be masked
    fn should_mask(&self, local_pos: Coords2d) -> bool;
}
