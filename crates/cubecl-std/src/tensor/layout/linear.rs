use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::{
    is_contiguous, is_contiguous_pitched,
    launch::{TypedView, TypedViewLaunch},
    layout::{
        Coords1d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand,
        permuted::{PermutedLayout, PermutedLayoutLaunch},
        strided::{StridedLayout, StridedLayoutLaunch},
        virtual_layout,
    },
    r#virtual::Read,
};

/// Maps a linear index based on line count to a potentially strided tensor. Only applies the
/// necessary level of striding, either none, only the last dim (for freshly allocated strided
/// tensors), or all dimensions.
///
/// `Layout` version of [index_offset_contiguous]
#[derive(CubeType, CubeLaunch, Clone)]
pub enum LinearLayout {
    /// Input is contiguous, no mapping
    Plain { len: u32 },
    /// Strided tensor, i.e. freshly allocated but not permuted
    Strided(StridedLayout),
    /// Permuted layout, tracks the entire shape/strides and not just the last dim
    Permuted(PermutedLayout),
}

impl<'a, R: Runtime> LinearLayoutArgs<'a, R> {
    pub fn from_shape_strides(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        strides: &[usize],
        line_size: &'a u8,
    ) -> Self {
        let rank = shape.len();
        if rank == 1 || is_contiguous(shape, strides) {
            let len = shape.iter().product::<usize>();
            let len = len / *line_size as usize;
            Self::Plain {
                len: ScalarArg::new(len as u32),
            }
        } else if is_contiguous_pitched(shape, strides) {
            Self::Strided(StridedLayoutLaunch::from_shape_strides(
                client, shape, strides, line_size,
            ))
        } else {
            Self::Permuted(PermutedLayoutLaunch::from_shape_strides(
                client, shape, strides, line_size,
            ))
        }
    }

    pub fn from_handle(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'a, R>,
        line_size: &'a u8,
    ) -> Self {
        Self::from_shape_strides(client, handle.shape, handle.strides, line_size)
    }
}

#[cube]
impl Layout for LinearLayout {
    type Coordinates = Coords1d;

    fn to_linear_pos(this: &Self, pos: Self::Coordinates) -> u32 {
        match this {
            LinearLayout::Plain { .. } => pos,
            LinearLayout::Strided(strided_layout) => strided_layout.to_linear_pos(pos),
            LinearLayout::Permuted(permuted_layout) => permuted_layout.to_linear_pos(pos),
        }
    }

    fn to_linear_pos_checked(this: &Self, pos: Self::Coordinates) -> (u32, bool) {
        match this {
            LinearLayout::Plain { len } => {
                let idx = this.to_linear_pos(pos);
                let in_bounds = pos < *len;
                (idx, in_bounds)
            }
            LinearLayout::Strided(strided_layout) => strided_layout.to_linear_pos_checked(pos),
            LinearLayout::Permuted(permuted_layout) => permuted_layout.to_linear_pos_checked(pos),
        }
    }

    fn shape(this: &Self) -> Self::Coordinates {
        match this {
            LinearLayout::Plain { len } => *len,
            LinearLayout::Strided(strided_layout) => strided_layout.shape(),
            LinearLayout::Permuted(permuted_layout) => permuted_layout.shape(),
        }
    }
}

virtual_layout!(LinearLayout, LinearLayoutExpand);

/// [TensorView] with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type LinearView<E, IO = Read> = TypedView<E, LinearLayout, IO>;
/// Launch type for [LinearTensorView].
pub type LinearViewLaunch<'a, R> = TypedViewLaunch<'a, LinearLayout, R>;

/// Create a linear tensor view from a handle and line size
pub fn linear_view<'a, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    handle: &'a TensorHandleRef<'a, R>,
    line_size: &'a u8,
) -> LinearViewLaunch<'a, R> {
    let len = handle.shape.iter().product::<usize>();
    let layout = LinearLayoutArgs::from_handle(client, handle, line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(handle.handle, len, *line_size, handle.elem_size)
    };
    LinearViewLaunch::new(buffer, layout)
}

pub fn linear_view_alias<'a, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    handle: &'a TensorHandleRef<'a, R>,
    line_size: &'a u8,
    pos: usize,
) -> LinearViewLaunch<'a, R> {
    let layout = LinearLayoutArgs::from_handle(client, handle, line_size);
    let buffer = ArrayArg::Alias { input_pos: pos };
    LinearViewLaunch::new(buffer, layout)
}
