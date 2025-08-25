use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::{
        index_offset_contiguous_fastdivmod,
        layout::{
            Coords1d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand,
            virtual_layout,
        },
    },
};

/// Layout for mapping heavily permuted tensors that can't be indexed as linear or 2D strided to a
/// linear index
#[derive(CubeType, CubeLaunch, Clone)]
pub struct PermutedLayout {
    shape: Sequence<FastDivmod>,
    strides: Sequence<u32>,
    len: u32,
    #[cube(comptime)]
    line_size: u8,
}

impl<'a, R: Runtime> PermutedLayoutLaunch<'a, R> {
    pub fn from_shape_strides(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        strides: &[usize],
        line_size: &'a u8,
    ) -> Self {
        let len = shape.iter().product::<usize>();

        let shape = SequenceArg {
            values: shape
                .iter()
                .map(|it| FastDivmodArgs::new(client, *it as u32))
                .collect(),
        };
        let strides = SequenceArg {
            values: strides
                .iter()
                .map(|it| ScalarArg::new(*it as u32))
                .collect(),
        };

        Self::new(shape, strides, ScalarArg::new(len as u32), line_size)
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
impl Layout for PermutedLayout {
    type Coordinates = Coords1d;

    fn to_linear_pos(this: &Self, pos: Self::Coordinates) -> u32 {
        index_offset_contiguous_fastdivmod(
            pos,
            &this.shape,
            &this.strides,
            comptime![this.line_size as u32],
        )
    }

    fn to_linear_pos_checked(this: &Self, pos: Self::Coordinates) -> (u32, bool) {
        let idx = this.to_linear_pos(pos);
        let in_bounds = pos < this.len;
        (idx, in_bounds)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        this.len
    }
}

virtual_layout!(PermutedLayout, PermutedLayoutExpand);
