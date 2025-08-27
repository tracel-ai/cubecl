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

    /// Transform a set of n-dimensional coordinates to a linear offset into the underlying storage
    /// (i.e. tensor, shared memory). This index should be adjusted for line size, so something like
    /// this would fetch the correct element in the conceptual tensor:
    /// ```ignore
    /// let idx = layout.to_linear_pos((1, 2));
    /// let value = tensor[idx];
    /// ```
    fn to_linear_pos(this: &Self, pos: Self::Coordinates) -> u32;
    /// Transform a set of n-dimensional coordinates to a linear offset into the underlying storage,
    /// and return whether the position is in bounds of this layout.
    /// See also [Layout::to_linear_pos]
    fn to_linear_pos_checked(this: &Self, pos: Self::Coordinates) -> (u32, bool);
    /// The shape of the conceptual tensor represented by this layout. Not necessarily the extent
    /// of the underlying storage, but only this view of it.
    fn shape(this: &Self) -> Self::Coordinates;
}

/// A layout transform, going from one set of coordinates to another. May be used for composite
/// layouts or to translate between coordinate systems for things like TMA loads.
#[cube]
pub trait LayoutTransform<From: Coordinates, To: Coordinates>:
    CubeType + Clone + Send + Sync + 'static
{
    fn transform_to(this: &Self, coords: From) -> To;
    fn transform_from(this: &Self, coords: To) -> From;
}

/// Dummy transform, no change to the coordinates. Can be used for 1 to 1 mappings.
#[derive(Clone, Copy)]
pub struct NoTransform;

impl CubeType for NoTransform {
    type ExpandType = Self;
}

impl IntoMut for NoTransform {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for NoTransform {}

#[cube]
impl<T: Coordinates> LayoutTransform<T, T> for NoTransform {
    fn transform_to(_this: &Self, coords: T) -> T {
        coords
    }

    fn transform_from(_this: &Self, coords: T) -> T {
        coords
    }
}

macro_rules! virtual_layout {
    ($ty: ident, $expand: ident) => {
        mod r#virtual {
            use super::*;
            use crate::tensor::layout::*;
            type L = $ty;
            type Coords = <L as Layout>::Coordinates;
            type CoordsExpand = <Coords as CubeType>::ExpandType;

            impl VirtualLayoutOperationsExpand<Coords> for $expand {
                fn __expand_to_linear_pos_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <u32 as CubeType>::ExpandType {
                    L::__expand_to_linear_pos(scope, self.clone(), pos)
                }
                fn __expand_to_linear_pos_checked_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <(u32, bool) as CubeType>::ExpandType {
                    L::__expand_to_linear_pos_checked(scope, self.clone(), pos)
                }
                fn __expand_shape_method(&self, scope: &mut Scope) -> CoordsExpand {
                    L::__expand_shape(scope, self.clone())
                }
            }

            #[cube]
            impl $ty {
                pub fn virt(self) -> VirtualLayout<Coords> {
                    VirtualLayout::new::<L>(self)
                }
            }
        }
    };
}

pub(crate) use virtual_layout;
