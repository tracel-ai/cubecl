use cubecl::prelude::*;
use cubecl_core as cubecl;

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
    shape: Sequence<FastDivmod>,
    strides: Sequence<u32>,
    len: u32,
    #[cube(comptime)]
    line_size: u8,
}

impl<'a, R: Runtime> PermutedLayoutLaunch<'a, R> {
    /// Create a new permuted layout for a possibly broadcast tensor, with a reference shape to be
    /// broadcast to.
    pub fn from_shape_strides(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        strides: &[usize],
        line_size: &'a u8,
    ) -> Self {
        let len = shape.iter().product::<usize>() / *line_size as usize;

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

    /// Create a new permuted layout for a possibly broadcast tensor, with a reference shape to be
    /// broadcast to.
    pub fn from_shapes_strides_ref(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        reference_shape: &[usize],
        strides: &[usize],
        line_size: &'a u8,
    ) -> Self {
        debug_assert!(
            shape.len() == reference_shape.len(),
            "Shape and reference should have the same rank"
        );
        debug_assert!(
            shape
                .iter()
                .zip(reference_shape)
                .all(|(s, r)| s == r || *s == 1),
            "Shape should be equal to reference or 1 on each dimension"
        );

        let strides: Vec<usize> = strides
            .iter()
            .zip(shape.iter().zip(reference_shape))
            .map(|(stride, (s, r))| if *s == *r { *stride } else { 0 })
            .collect();

        Self::from_shape_strides(client, reference_shape, &strides, line_size)
    }

    pub fn from_handles_ref(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        reference_handle: &TensorHandleRef<'_, R>,
        line_size: &'a u8,
    ) -> Self {
        Self::from_shapes_strides_ref(
            client,
            handle.shape,
            reference_handle.shape,
            handle.strides,
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
impl Layout for PermutedLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> u32 {
        index_offset_contiguous_fastdivmod(
            pos,
            &self.shape,
            &self.strides,
            comptime![self.line_size as u32],
        )
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
