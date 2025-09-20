use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};

use crate::tensor::layout::{Coordinates, Layout, LayoutExpand};

/// A layout containing a sub-slice of the inner layout.
#[allow(unused)]
#[derive(CubeType)]
pub struct SliceLayout<C: Coordinates> {
    offset: C,
    size: C,
    #[cube(comptime)]
    checked: bool,
}

#[cube]
impl<C: Coordinates> SliceLayout<C> {
    /// Create a new slice layout.
    /// `checked` determines whether bounds should be checked, or simply treated as always in bounds.
    pub fn new(start: C, size: C, #[comptime] checked: bool) -> Self {
        SliceLayout::<C> {
            offset: start,
            size,
            checked,
        }
    }

    fn offset(&self) -> C {
        intrinsic! {|_| self.offset.clone() }
    }

    fn size(&self) -> C {
        intrinsic! {|_| self.size.clone() }
    }
}

#[cube]
impl<C: Coordinates> Layout for SliceLayout<C> {
    type Coordinates = C;
    type SourceCoordinates = C;

    fn shape(&self) -> Self::Coordinates {
        self.size()
    }

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        C::add(self.offset(), pos)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        if comptime![self.checked] {
            C::is_in_bounds(&pos, &self.size)
        } else {
            true.runtime()
        }
    }

    fn to_source_shape(&self, shape: Self::Coordinates) -> Self::SourceCoordinates {
        shape
    }

    #[allow(unused)]
    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        intrinsic!(|scope| {
            let in_bounds = self
                .clone()
                .__expand_is_in_bounds_method(scope, pos.clone());
            let pos = self.__expand_to_source_pos_method(scope, pos);
            (pos, in_bounds)
        })
    }
}
