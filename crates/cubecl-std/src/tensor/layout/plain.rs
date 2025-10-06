use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{Coords1d, Layout, LayoutExpand};

/// Layout for contiguous tensors.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct PlainLayout {
    len: u32,
}

#[cube]
impl PlainLayout {
    pub fn new(len: u32) -> Self {
        PlainLayout { len }
    }
}

impl<'a, R: Runtime> PlainLayoutLaunch<'a, R> {
    pub fn from_shape(shape: &[usize], line_size: u8) -> Self {
        let len = shape.iter().product::<usize>();
        let len = len / line_size as usize;
        Self::new(ScalarArg::new(len as u32))
    }

    pub fn from_handle(handle: &TensorHandleRef<'_, R>, line_size: u8) -> Self {
        Self::from_shape(handle.shape, line_size)
    }
}

#[cube]
impl Layout for PlainLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> u32 {
        pos
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (u32, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.len
    }
}
