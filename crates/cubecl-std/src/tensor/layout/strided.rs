use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use crate::{
    FastDivmod,
    tensor::{
        launch::{BufferArg, ViewLayoutLaunchArg},
        layout::{Coords1d, Layout, LayoutExpand},
    },
};

/// Layout for tensors strided only on the last dimension, i.e. freshly allocated ones. Treats the
/// tensor as 2D for the purposes of indexing, with the remaining dimensions being collapsed into
/// a single contiguous one
#[derive(CubeType, Clone)]
pub struct StridedLayout {
    shape: FastDivmod<usize>,
    stride: usize,
    len: usize,
    #[cube(comptime)]
    vector_size: VectorSize,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct StridedLayoutCompilationArg {
    shape: <FastDivmod<usize> as LaunchArg>::CompilationArg,
}

impl ViewLayoutLaunchArg for StridedLayout {
    type RuntimeArg<R: Runtime> = ();
    type CompilationArg = StridedLayoutCompilationArg;

    fn register<R: Runtime, B: BufferArg>(
        _: Self::RuntimeArg<R>,
        buffer: &B,
        ty: Type,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let shape = buffer.shape();
        let strides = buffer.strides();
        let rank = shape.len();
        let len = shape.iter().product::<usize>() / ty.vector_size();

        let shape = <FastDivmod<usize> as LaunchArg>::register(shape[rank - 1], launcher);
        <usize as LaunchArg>::register(strides[rank - 2], launcher);
        <usize as LaunchArg>::register(len, launcher);
        StridedLayoutCompilationArg { shape }
    }

    fn expand(
        arg: &Self::CompilationArg,
        ty: Type,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        StridedLayoutExpand {
            shape: <FastDivmod<usize> as LaunchArg>::expand(&arg.shape, builder),
            stride: <usize as LaunchArg>::expand(&(), builder),
            len: <usize as LaunchArg>::expand(&(), builder),
            vector_size: ty.vector_size(),
        }
    }
}

#[cube]
impl Layout for StridedLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        let offset_abs = pos * self.vector_size;
        let (y, x) = self.shape.div_mod(offset_abs);
        let offset = y * self.stride + x;
        offset / self.vector_size
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
