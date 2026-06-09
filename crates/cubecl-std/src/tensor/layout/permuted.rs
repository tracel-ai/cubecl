use cubecl::prelude::*;
use cubecl_core::{self as cubecl, ir::VectorSize, zspace::Shape};

use crate::{
    FastDivmod,
    tensor::{
        index_offset_contiguous_fastdivmod,
        launch::{MemoryArg, ViewLayoutLaunchArg},
        layout::{Coords1d, Layout, LayoutExpand},
    },
};

/// Layout for mapping heavily permuted tensors that can't be indexed as linear or 2D strided to a
/// linear index
#[derive(CubeType, Clone)]
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

#[derive(Default)]
pub struct PermutedLayoutLaunch {
    reference_shape: Option<Shape>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct PermutedLayoutCompilationArg {
    shape: <Sequence<FastDivmod<usize>> as LaunchArg>::CompilationArg,
    strides: <Sequence<usize> as LaunchArg>::CompilationArg,
}

impl ViewLayoutLaunchArg for PermutedLayout {
    type RuntimeArg<R: Runtime> = PermutedLayoutLaunch;
    type CompilationArg = PermutedLayoutCompilationArg;

    fn register<R: Runtime, B: MemoryArg>(
        arg: Self::RuntimeArg<R>,
        buffer: &B,
        ty: Type,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let shape = buffer.shape();
        let strides = buffer.strides();
        let (shape, strides, len) = match arg.reference_shape {
            Some(reference_shape) => {
                let len = reference_shape.len();
                let strides = strides_ref(shape, &reference_shape, strides);
                (reference_shape.iter().copied().collect(), strides, len)
            }
            None => (
                shape.iter().copied().collect(),
                strides.iter().copied().collect(),
                buffer.len(),
            ),
        };
        let len = len / ty.vector_size();
        let shape = <Sequence<FastDivmod<usize>> as LaunchArg>::register(shape, launcher);
        let strides = <Sequence<usize> as LaunchArg>::register(strides, launcher);
        <usize as LaunchArg>::register(len, launcher);
        PermutedLayoutCompilationArg { shape, strides }
    }

    fn expand(
        arg: &Self::CompilationArg,
        ty: Type,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        PermutedLayoutExpand {
            shape: <Sequence<FastDivmod<usize>> as LaunchArg>::expand(&arg.shape, builder),
            strides: <Sequence<usize> as LaunchArg>::expand(&arg.strides, builder),
            len: <usize as LaunchArg>::expand(&(), builder),
            vector_size: ty.vector_size(),
        }
    }
}

fn strides_ref<R: Runtime>(
    shape: &[usize],
    reference_shape: &[usize],
    strides: &[usize],
) -> SequenceArg<R, usize> {
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

    strides
        .iter()
        .zip(shape.iter().zip(reference_shape.iter()))
        .map(|(stride, (s, r))| if *s == *r { *stride } else { 0 })
        .collect()
}

impl PermutedLayoutLaunch {
    /// Create a new permuted layout without a reference shape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new permuted layout for a possibly broadcast tensor, with a reference shape to be
    /// broadcast to.
    pub fn from_reference_shape(reference_shape: Shape) -> Self {
        Self {
            reference_shape: Some(reference_shape),
        }
    }

    pub fn from_reference_handle<R: Runtime>(reference_handle: TensorBinding<R>) -> Self {
        Self::from_reference_shape(reference_handle.shape)
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
