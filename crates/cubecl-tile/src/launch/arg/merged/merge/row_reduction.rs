//! How a row reduces to one value: [`RowReduction`].

use cubecl::prelude::*;

/// How a row reduces to one value: each cell mapped and the row's cells summed, then the sum mapped
/// once, as an RMS norm's factor is of its row's squares.
#[cube]
pub trait RowReduction: Send + Sync + 'static {
    /// What the map of the sum reads besides the sum.
    type Param: LaunchArg + CubeType;

    /// What each cell of a vector adds to its row's sum.
    fn cells<E: Numeric, N: Size>(value: Vector<E, N>) -> Vector<f32, N>;

    /// The row's value, from its sum.
    fn of_sum(sum: f32, param: &Self::Param) -> f32;
}
