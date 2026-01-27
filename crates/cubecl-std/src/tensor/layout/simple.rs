use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{Coords1d, Layout, LayoutExpand};

/// Layout for contiguous tensors, indexed in individual elements.
/// Differs from `PlainLayout` because `PlainLayout` expects line indices, not element indices.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct SimpleLayout {
    len: usize,
    #[cube(comptime)]
    line_size: LineSize,
}

#[cube]
impl SimpleLayout {
    /// Create a new simple layout with a length and line size.
    ///
    /// # Note
    /// Length should be in elements, not lines!
    pub fn new(len: usize, #[comptime] line_size: LineSize) -> Self {
        SimpleLayout { len, line_size }
    }
}

impl<'a, R: Runtime> SimpleLayoutLaunch<'a, R> {
    pub fn from_shape(shape: &[usize], line_size: LineSize) -> Self {
        let len = shape.iter().product::<usize>();
        Self::new(ScalarArg::new(len), line_size)
    }

    pub fn from_handle(handle: &TensorHandleRef<'_, R>, line_size: LineSize) -> Self {
        Self::from_shape(handle.shape, line_size)
    }
}

#[cube]
impl Layout for SimpleLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        pos / self.line_size
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
