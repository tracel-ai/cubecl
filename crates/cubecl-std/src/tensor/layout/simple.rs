use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{Coords1d, Layout, LayoutExpand};

/// Layout for contiguous tensors, indexed in individual elements.
/// Differs from `PlainLayout` because `PlainLayout` expects line indices, not element indices.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct SimpleLayout {
    len: u32,
    #[cube(comptime)]
    line_size: u8,
}

#[cube]
impl SimpleLayout {
    pub fn new(len: u32, #[comptime] line_size: u32) -> Self {
        SimpleLayout {
            len,
            line_size: comptime![line_size as u8],
        }
    }
}

impl<'a, R: Runtime> SimpleLayoutLaunch<'a, R> {
    pub fn from_shape(shape: &[usize], line_size: &'a u8) -> Self {
        let len = shape.iter().product::<usize>();
        Self::new(ScalarArg::new(len as u32), line_size)
    }

    pub fn from_handle(handle: &TensorHandleRef<'_, R>, line_size: &'a u8) -> Self {
        Self::from_shape(handle.shape, line_size)
    }
}

#[cube]
impl Layout for SimpleLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> u32 {
        pos / comptime![self.line_size as u32]
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (u32, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.len
    }

    fn to_source_shape(&self, shape: Self::Coordinates) -> Self::SourceCoordinates {
        shape / comptime![self.line_size as u32]
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.len
    }
}
