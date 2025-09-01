use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{
        Coords1d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand, virtual_layout,
    },
};

/// Layout for tensors strided only on the last dimension, i.e. freshly allocated ones. Treats the
/// tensor as 2D for the purposes of indexing, with the remaining dimensions being collapsed into
/// a single contiguous one
#[derive(CubeType, CubeLaunch, Clone)]
pub struct StridedLayout {
    shape: FastDivmod,
    stride: u32,
    len: u32,
    #[cube(comptime)]
    line_size: u8,
}

impl<'a, R: Runtime> StridedLayoutLaunch<'a, R> {
    pub fn from_shape_strides(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        strides: &[usize],
        line_size: &'a u8,
    ) -> Self {
        let rank = shape.len();
        let len = shape.iter().product::<usize>();
        Self::new(
            FastDivmodArgs::new(client, shape[rank - 1] as u32),
            ScalarArg::new(strides[rank - 2] as u32),
            ScalarArg::new(len as u32),
            line_size,
        )
    }

    pub fn from_handle(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        line_size: &'a u8,
    ) -> Self {
        Self::from_shape_strides(client, handle.shape, handle.strides, line_size)
    }
}

#[cube]
impl Layout for StridedLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(this: &Self, pos: Self::Coordinates) -> u32 {
        let offset_abs = pos * comptime![this.line_size as u32];
        let (y, x) = this.shape.div_mod(offset_abs);
        y * this.stride + x
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

virtual_layout!(StridedLayout, StridedLayoutExpand);
