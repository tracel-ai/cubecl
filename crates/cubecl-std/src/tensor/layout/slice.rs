use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};

use crate::tensor::layout::{Coordinates, Layout, LayoutExpand};

#[allow(unused)]
#[derive(CubeType)]
pub struct SliceLayout<C: Coordinates> {
    offset: C,
    size: C,
}

#[cube]
impl<C: Coordinates> SliceLayout<C> {
    /// Create a new slice layout.
    pub fn new(start: C, size: C) -> Self {
        SliceLayout::<C> {
            offset: start,
            size,
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
        C::is_in_bounds(pos, self.size())
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
