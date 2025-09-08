use cubecl::prelude::*;
use cubecl_core::{self as cubecl, unexpanded};

use crate::tensor::{
    is_contiguous, is_contiguous_pitched,
    launch::{TypedView, TypedViewLaunch},
    layout::{
        Coords1d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand,
        permuted::{PermutedLayout, PermutedLayoutLaunch},
        plain::{PlainLayout, PlainLayoutLaunch},
        strided::{StridedLayout, StridedLayoutLaunch},
        virtual_layout,
    },
};

/// Maps a linear index based on line count to a potentially strided tensor. Only applies the
/// necessary level of striding, either none, only the last dim (for freshly allocated strided
/// tensors), or all dimensions.
///
/// Treats indices as the line index, with the shape being adjusted for line size.
///
/// `Layout` version of [index_offset_contiguous]
#[derive(CubeType, CubeLaunch, Clone)]
pub enum LinearLayout {
    /// Input is contiguous, no mapping
    Plain(PlainLayout),
    /// Strided tensor, i.e. freshly allocated but not permuted
    Strided(StridedLayout),
    /// Permuted layout, tracks the entire shape/strides and not just the last dim
    Permuted(PermutedLayout),
}

impl LinearLayout {
    fn inner(&self) -> &dyn VirtualLayoutOperations<Coords1d, Coords1d> {
        unexpanded!()
    }
}

impl LinearLayoutExpand {
    fn __expand_inner_method(
        &self,
        _scope: &mut Scope,
    ) -> &dyn VirtualLayoutOperationsExpand<Coords1d, Coords1d> {
        match self {
            LinearLayoutExpand::Plain(layout) => layout,
            LinearLayoutExpand::Strided(layout) => layout,
            LinearLayoutExpand::Permuted(layout) => layout,
        }
    }
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
            Self::Plain(PlainLayoutLaunch::from_shape(shape, line_size))
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
    type SourceCoordinates = Coords1d;

    fn to_source_pos(this: &Self, pos: Self::Coordinates) -> u32 {
        this.inner().to_source_pos(pos)
    }

    fn to_source_pos_checked(this: &Self, pos: Self::Coordinates) -> (u32, bool) {
        (this.to_source_pos(pos), this.is_in_bounds(pos))
    }

    fn shape(this: &Self) -> Self::Coordinates {
        this.inner().shape()
    }

    fn is_in_bounds(this: &Self, pos: Self::Coordinates) -> bool {
        this.inner().is_in_bounds(pos)
    }
}

virtual_layout!(LinearLayout, LinearLayoutExpand);

/// [TensorView] with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type LinearView<E, IO = ReadOnly> = TypedView<E, LinearLayout, IO>;
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
