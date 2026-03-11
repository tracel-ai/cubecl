use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    ir::VectorSize,
    zspace::{Shape, Strides},
};

use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::{
        index_offset_contiguous_fastdivmod,
        layout::{Coords1d, Layout, LayoutExpand},
    },
};

/// Layout for mapping heavily permuted tensors that can't be indexed as linear or 2D strided to a
/// linear index
#[derive(CubeType, CubeLaunch, Clone)]
pub struct PermutedLayout {
    shape: Sequence<FastDivmod<usize>>,
    strides: Sequence<usize>,
    len: usize,
    #[cube(comptime)]
    vector_size: VectorSize,
}

#[cube]
impl PermutedLayout {
    pub fn new(
        shape: Sequence<FastDivmod<usize>>,
        strides: Sequence<usize>,
        len: usize,
        #[comptime] vector_size: VectorSize,
    ) -> Self {
        PermutedLayout {
            shape,
            strides,
            len,
            vector_size,
        }
    }
}

impl<R: Runtime> PermutedLayoutLaunch<R> {
    /// Create a new permuted layout for a possibly broadcast tensor, with a reference shape to be
    /// broadcast to.
    pub fn from_shape_strides(
        client: &ComputeClient<R>,
        shape: &Shape,
        strides: &Strides,
        vector_size: VectorSize,
    ) -> Self {
        let len = shape.iter().product::<usize>() / vector_size;

        let shape = shape
            .iter()
            .map(|it| FastDivmodArgs::<usize>::new(client, *it))
            .collect();
        let strides = strides.iter().copied().collect();

        Self::new(shape, strides, len, vector_size)
    }

    /// Create a new permuted layout for a possibly broadcast tensor, with a reference shape to be
    /// broadcast to.
    pub fn from_shapes_strides_ref(
        client: &ComputeClient<R>,
        shape: &Shape,
        reference_shape: &Shape,
        strides: &Strides,
        vector_size: VectorSize,
    ) -> Self {
        debug_assert!(
            shape.len() == reference_shape.len(),
            "Shape and reference should have the same rank"
        );
        debug_assert!(
            shape
                .iter()
                .zip(reference_shape.iter())
                .all(|(s, r)| s == r || *s == 1),
            "Shape should be equal to reference or 1 on each dimension"
        );

        let strides: Strides = strides
            .iter()
            .zip(shape.iter().zip(reference_shape.iter()))
            .map(|(stride, (s, r))| if *s == *r { *stride } else { 0 })
            .collect();

        Self::from_shape_strides(client, reference_shape, &strides, vector_size)
    }

    pub fn from_handles_ref(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        reference_handle: TensorBinding<R>,
        vector_size: VectorSize,
    ) -> Self {
        Self::from_shapes_strides_ref(
            client,
            &handle.shape,
            &reference_handle.shape,
            &handle.strides,
            vector_size,
        )
    }

    pub fn from_handle(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        vector_size: VectorSize,
    ) -> Self {
        Self::from_shape_strides(client, &handle.shape, &handle.strides, vector_size)
    }
}

#[cube]
impl Layout for PermutedLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        index_offset_contiguous_fastdivmod(pos, &self.shape, &self.strides, self.vector_size)
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
