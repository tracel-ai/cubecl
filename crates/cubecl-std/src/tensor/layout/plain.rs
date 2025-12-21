use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{Coords1d, Layout, LayoutExpand};

/// Layout for contiguous tensors.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct PlainLayout {
    len: usize,
}

#[cube]
impl PlainLayout {
    pub fn new(len: usize) -> Self {
        PlainLayout { len }
    }
}

impl<'a, R: Runtime> PlainLayoutLaunch<'a, R> {
    pub fn from_shape(shape: &[usize], line_size: LineSize) -> Self {
        let len = shape.iter().product::<usize>();
        let len = len / line_size;
        Self::new(ScalarArg::new(len))
    }

    pub fn from_handle(handle: &TensorHandleRef<'_, R>, line_size: LineSize) -> Self {
        Self::from_shape(handle.shape, line_size)
    }
}

#[cube]
impl Layout for PlainLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        pos
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.len
    }
}
