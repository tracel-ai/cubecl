use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use super::Coordinates;

/// A layout that represents the mapping from a conceptual multi-dimensional tensor to a linear
/// storage. Some layouts may be transformers meant to be composed with others (i.e. swizzling),
/// others will represent the actual underlying structure of the data.
///
/// The expand type should also implement
/// [VirtualLayoutOperationsExpand](crate::tensor::layout::VirtualLayoutOperationsExpand) to be
/// compatible with [VirtualLayout](crate::tensor::layout::VirtualLayout).
#[cube]
pub trait Layout: CubeType + Clone + Send + Sync + 'static {
    /// The coordinate type used by the conceptual tensor represented by this layout, i.e.
    /// `(u32, u32, u32)` for a fixed-rank 3D tensor.
    /// This does not have to match the rank of the underlying storage (if applicable).
    /// It's only how the tensor is interpreted (viewed) by the code.
    type Coordinates: Coordinates;
    /// The coordinate type used by the inner storage wrapped in this layout, i.e. `u32` for
    /// `Array`, or `(u32, u32)` for a 2D view.
    type SourceCoordinates: Coordinates;

    /// Transform a set of n-dimensional coordinates to an offset into the underlying storage
    /// (i.e. tensor, shared memory). Implementations on concrete storage should account for line
    /// size - coordinates are given in elements, not lines.
    fn to_source_pos(this: &Self, pos: Self::Coordinates) -> Self::SourceCoordinates;
    /// Transform a set of n-dimensional coordinates to an offset into the underlying storage,
    /// and return whether the position is in bounds of this layout.
    /// See also [Layout::to_source_pos]
    fn to_source_pos_checked(
        this: &Self,
        pos: Self::Coordinates,
    ) -> (Self::SourceCoordinates, bool);
    /// The shape of the conceptual tensor represented by this layout. Not necessarily the extent
    /// of the underlying storage, but only this view of it.
    fn shape(this: &Self) -> Self::Coordinates;
    fn is_in_bounds(this: &Self, pos: Self::Coordinates) -> bool;
}

macro_rules! virtual_layout {
    ($ty: ident, $expand: ident) => {
        mod r#virtual {
            use super::*;
            use crate::tensor::layout::*;
            type L = $ty;
            type Coords = <L as Layout>::Coordinates;
            type SourceCoords = <L as Layout>::SourceCoordinates;
            type CoordsExpand = <Coords as CubeType>::ExpandType;

            impl VirtualLayoutOperationsExpand<Coords, SourceCoords> for $expand {
                fn __expand_to_source_pos_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <SourceCoords as CubeType>::ExpandType {
                    L::__expand_to_source_pos(scope, self.clone(), pos)
                }
                fn __expand_to_source_pos_checked_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <(SourceCoords, bool) as CubeType>::ExpandType {
                    L::__expand_to_source_pos_checked(scope, self.clone(), pos)
                }
                fn __expand_shape_method(&self, scope: &mut Scope) -> CoordsExpand {
                    L::__expand_shape(scope, self.clone())
                }
                fn __expand_is_in_bounds_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> ExpandElementTyped<bool> {
                    L::__expand_is_in_bounds(scope, self.clone(), pos)
                }
            }

            #[cube]
            impl $ty {
                pub fn virt(self) -> VirtualLayout<Coords, SourceCoords> {
                    VirtualLayout::new::<L>(self)
                }
            }
        }
    };
}

pub(crate) use virtual_layout;
