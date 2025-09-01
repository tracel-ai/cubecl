use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{
    Coords1d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand, virtual_layout,
};

/// Layout for contiguous tensors.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct PlainLayout {
    len: u32,
    #[cube(comptime)]
    line_size: u8,
}

impl<'a, R: Runtime> PlainLayoutLaunch<'a, R> {
    pub fn from_shape(shape: &[usize], line_size: &'a u8) -> Self {
        let len = shape.iter().product::<usize>();
        Self::new(ScalarArg::new(len as u32), line_size)
    }

    pub fn from_handle(handle: &TensorHandleRef<'_, R>, line_size: &'a u8) -> Self {
        Self::from_shape(handle.shape, line_size)
    }
}

#[cube]
impl Layout for PlainLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(this: &Self, pos: Self::Coordinates) -> u32 {
        pos * comptime!(this.line_size as u32)
    }

    fn to_source_pos_checked(this: &Self, pos: Self::Coordinates) -> (u32, bool) {
        (this.to_source_pos(pos), this.is_in_bounds(pos))
    }

    fn shape(this: &Self) -> Self::Coordinates {
        this.len
    }

    fn is_in_bounds(this: &Self, pos: Self::Coordinates) -> bool {
        pos < this.len
    }
}

virtual_layout!(PlainLayout, PlainLayoutExpand);
