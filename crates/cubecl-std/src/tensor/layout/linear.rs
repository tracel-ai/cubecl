use alloc::rc::Rc;

use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, unexpanded,
    zspace::{Shape, Strides},
};

use crate::tensor::{
    View, is_contiguous, is_contiguous_pitched,
    launch::ViewArg,
    layout::{
        Coords1d, Layout, LayoutExpand, VirtualLayoutOperationsExpand,
        permuted::{PermutedLayout, PermutedLayoutLaunch},
        plain::{PlainLayout, PlainLayoutLaunch},
        strided::{StridedLayout, StridedLayoutLaunch},
    },
};

/// Maps a linear index based on line count to a potentially strided tensor. Only applies the
/// necessary level of striding, either none, only the last dim (for freshly allocated strided
/// tensors), or all dimensions.
///
/// Treats indices as the line index, with the shape being adjusted for line size.
///
/// `Layout` version of [`crate::tensor::contiguous::index_offset_contiguous()`]
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
    fn inner(&self) -> &PlainLayout {
        unexpanded!()
    }
}

impl LinearLayoutExpand {
    fn __expand_inner_method(
        self,
        _scope: &mut Scope,
    ) -> Rc<dyn VirtualLayoutOperationsExpand<Coords1d, Coords1d>> {
        match self {
            LinearLayoutExpand::Plain(layout) => Rc::new(layout),
            LinearLayoutExpand::Strided(layout) => Rc::new(layout),
            LinearLayoutExpand::Permuted(layout) => Rc::new(layout),
        }
    }
}

impl<'a, R: Runtime> LinearLayoutArgs<'a, R> {
    /// Construct a linear layout from shapes, strides and line size of the tensor
    pub fn from_shape_strides(
        client: &ComputeClient<R>,
        shape: &Shape,
        strides: &Strides,
        line_size: LineSize,
    ) -> Self {
        if is_contiguous(shape, strides) {
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

    /// Construct a possibly broadcast linear layout from shapes/strides and a reference shape
    pub fn from_shape_strides_with_reference(
        client: &ComputeClient<R>,
        shape: &Shape,
        reference_shape: &Shape,
        strides: &Strides,
        line_size: LineSize,
    ) -> Self {
        if shape != reference_shape {
            // Broadcast layouts are always treated as permuted
            Self::Permuted(PermutedLayoutLaunch::from_shapes_strides_ref(
                client,
                shape,
                reference_shape,
                strides,
                line_size,
            ))
        } else {
            Self::from_shape_strides(client, shape, strides, line_size)
        }
    }

    /// Construct a linear layout from a tensor handle
    pub fn from_handle(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        line_size: LineSize,
    ) -> Self {
        Self::from_shape_strides(client, &handle.shape, &handle.strides, line_size)
    }

    /// Construct a possibly broadcast linear layout from a tensor handle and reference handle
    pub fn from_handle_with_reference(
        client: &ComputeClient<R>,
        handle: &TensorBinding<R>,
        reference: TensorBinding<R>,
        line_size: LineSize,
    ) -> Self {
        Self::from_shape_strides_with_reference(
            client,
            &handle.shape,
            &reference.shape,
            &handle.strides,
            line_size,
        )
    }
}

#[cube]
impl Layout for LinearLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        self.inner().to_source_pos(pos)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.inner().shape()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.inner().is_in_bounds(pos)
    }
}

/// [`View`] with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type LinearView<E, IO = ReadOnly> = View<E, Coords1d, IO>;
/// Launch type for [`LinearView`].
pub type LinearViewLaunch<'a, R> = ViewArg<'a, Coords1d, R>;

/// Create a linear tensor view from a handle and line size
pub fn linear_view<'a, R: Runtime>(
    client: &ComputeClient<R>,
    handle: TensorBinding<R>,
    line_size: LineSize,
) -> LinearViewLaunch<'a, R> {
    let len = handle.shape.iter().product::<usize>();
    let layout = LinearLayoutArgs::from_handle(client, handle.clone(), line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_binding(handle.handle, len, line_size, handle.elem_size)
    };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

/// Create a possibly broadcast linear tensor view from a handle, reference handle and line size
pub fn linear_view_with_reference<'a, R: Runtime>(
    client: &ComputeClient<R>,
    handle: TensorBinding<R>,
    reference: TensorBinding<R>,
    line_size: LineSize,
) -> LinearViewLaunch<'a, R> {
    let len = handle.shape.iter().product::<usize>();
    let layout =
        LinearLayoutArgs::from_handle_with_reference(client, &handle, reference, line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_binding(handle.handle, len, line_size, handle.elem_size)
    };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

pub fn linear_view_alias<'a, R: Runtime>(
    client: &ComputeClient<R>,
    handle: TensorBinding<R>,
    line_size: LineSize,
    pos: usize,
) -> LinearViewLaunch<'a, R> {
    let layout = LinearLayoutArgs::from_handle(client, handle, line_size);
    let buffer = ArrayArg::Alias { input_pos: pos };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}
