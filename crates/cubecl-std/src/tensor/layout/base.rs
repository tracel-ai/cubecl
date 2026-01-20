use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use super::Coordinates;

/// A layout that represents the mapping from a conceptual multi-dimensional tensor to a linear
/// storage. Some layouts may be transformers meant to be composed with others (i.e. swizzling),
/// others will represent the actual underlying structure of the data.
#[cube(expand_base_traits = "Clone")]
pub trait Layout {
    /// The coordinate type used by the conceptual tensor represented by this layout, i.e.
    /// `(u32, u32, u32)` for a fixed-rank 3D tensor.
    /// This does not have to match the rank of the underlying storage (if applicable).
    /// It's only how the tensor is interpreted (viewed) by the code.
    type Coordinates: Coordinates;
    /// The coordinate type used by the inner storage wrapped in this layout, i.e. `u32` for
    /// `Array`, or `(u32, u32)` for a 2D view.
    type SourceCoordinates: Coordinates;

    /// Transform a set of n-dimensional coordinates to a source coordinate space.
    /// It is recommended to use absolute positions here, and handle the translation into lines
    /// at the lowest level (global memory layout).
    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates;
    /// Transform a set of n-dimensional coordinates to an offset into the underlying storage,
    /// and return whether the position is in bounds of this layout.
    /// See also [`Layout::to_source_pos`]
    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool);
    /// The shape of the conceptual tensor represented by this layout. Not necessarily the extent
    /// of the underlying storage, but only this view of it.
    fn shape(&self) -> Self::Coordinates;
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool;
}
