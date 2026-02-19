use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    zspace::{Shape, Strides},
};

use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{Coords1d, Layout, LayoutExpand},
};

/// Layout for tensors strided only on the last dimension, i.e. freshly allocated ones. Treats the
/// tensor as 2D for the purposes of indexing, with the remaining dimensions being collapsed into
/// a single contiguous one
#[derive(CubeType, CubeLaunch, Clone)]
pub struct StridedLayout {
    shape: FastDivmod<usize>,
    stride: usize,
    len: usize,
    #[cube(comptime)]
    line_size: LineSize,
}

impl<'a, R: Runtime> StridedLayoutLaunch<'a, R> {
    pub fn from_shape_strides(
        client: &ComputeClient<R>,
        shape: &Shape,
        strides: &Strides,
        line_size: LineSize,
    ) -> Self {
        let rank = shape.len();
        let len = shape.iter().product::<usize>() / line_size;
        Self::new(
            FastDivmodArgs::<usize>::new(client, shape[rank - 1]),
            ScalarArg::new(strides[rank - 2]),
            ScalarArg::new(len),
            line_size,
        )
    }

    pub fn from_handle(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        line_size: LineSize,
    ) -> Self {
        Self::from_shape_strides(client, &handle.shape, &handle.strides, line_size)
    }
}

#[cube]
impl Layout for StridedLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        let offset_abs = pos * self.line_size;
        let (y, x) = self.shape.div_mod(offset_abs);
        let offset = y * self.stride + x;
        offset / self.line_size
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
