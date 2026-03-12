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

/// Maps a linear index based on vector count to a potentially strided tensor. Only applies the
/// necessary level of striding, either none, only the last dim (for freshly allocated strided
/// tensors), or all dimensions.
///
/// Treats indices as the vector index, with the shape being adjusted for vector size.
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

impl<R: Runtime> LinearLayoutArgs<R> {
    /// Construct a linear layout from shapes, strides and vector size of the tensor
    pub fn from_shape_strides(
        client: &ComputeClient<R>,
        shape: &Shape,
        strides: &Strides,
        vector_size: VectorSize,
    ) -> Self {
        if is_contiguous(shape, strides) {
            Self::Plain(PlainLayoutLaunch::from_shape(shape, vector_size))
        } else if is_contiguous_pitched(shape, strides) {
            Self::Strided(StridedLayoutLaunch::from_shape_strides(
                client,
                shape,
                strides,
                vector_size,
            ))
        } else {
            Self::Permuted(PermutedLayoutLaunch::from_shape_strides(
                client,
                shape,
                strides,
                vector_size,
            ))
        }
    }

    /// Construct a possibly broadcast linear layout from shapes/strides and a reference shape
    pub fn from_shape_strides_with_reference(
        client: &ComputeClient<R>,
        shape: &Shape,
        reference_shape: &Shape,
        strides: &Strides,
        vector_size: VectorSize,
    ) -> Self {
        if shape != reference_shape {
            // Broadcast layouts are always treated as permuted
            Self::Permuted(PermutedLayoutLaunch::from_shapes_strides_ref(
                client,
                shape,
                reference_shape,
                strides,
                vector_size,
            ))
        } else {
            Self::from_shape_strides(client, shape, strides, vector_size)
        }
    }

    /// Construct a linear layout from a tensor handle
    pub fn from_handle(
        client: &ComputeClient<R>,
        handle: &TensorBinding<R>,
        vector_size: VectorSize,
    ) -> Self {
        Self::from_shape_strides(client, &handle.shape, &handle.strides, vector_size)
    }

    /// Construct a possibly broadcast linear layout from a tensor handle and reference handle
    pub fn from_handle_with_reference(
        client: &ComputeClient<R>,
        handle: &TensorBinding<R>,
        reference: TensorBinding<R>,
        vector_size: VectorSize,
    ) -> Self {
        Self::from_shape_strides_with_reference(
            client,
            &handle.shape,
            &reference.shape,
            &handle.strides,
            vector_size,
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
pub type LinearViewLaunch<R> = ViewArg<Coords1d, R>;

/// Create a linear tensor view from a handle and vector size
pub fn linear_view<R: Runtime>(
    client: &ComputeClient<R>,
    handle: TensorBinding<R>,
    vector_size: VectorSize,
) -> LinearViewLaunch<R> {
    let len = handle.shape.iter().product::<usize>();
    let layout = LinearLayoutArgs::from_handle(client, &handle, vector_size);
    let buffer = unsafe { ArrayArg::from_raw_parts_binding(handle.handle, len) };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

/// Create a possibly broadcast linear tensor view from a handle, reference handle and vector size
pub fn linear_view_with_reference<R: Runtime>(
    client: &ComputeClient<R>,
    handle: TensorBinding<R>,
    reference: TensorBinding<R>,
    vector_size: VectorSize,
) -> LinearViewLaunch<R> {
    let len = handle.shape.iter().product::<usize>();
    let layout =
        LinearLayoutArgs::from_handle_with_reference(client, &handle, reference, vector_size);
    let buffer = unsafe { ArrayArg::from_raw_parts_binding(handle.handle, len) };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

pub fn linear_view_alias<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &TensorBinding<R>,
    vector_size: VectorSize,
    pos: usize,
) -> LinearViewLaunch<R> {
    let layout = LinearLayoutArgs::from_handle(client, handle, vector_size);
    let buffer = ArrayArg::Alias { input_pos: pos };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}
