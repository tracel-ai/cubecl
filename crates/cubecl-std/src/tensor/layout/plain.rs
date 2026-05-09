use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use crate::tensor::{
    launch::{MemoryArg, ViewLayoutLaunchArg},
    layout::{Coords1d, Layout, LayoutExpand},
};

/// Layout for contiguous tensors.
#[derive(CubeType, Clone)]
pub struct PlainLayout {
    len: usize,
}

#[cube]
impl PlainLayout {
    pub fn new(len: usize) -> Self {
        PlainLayout { len }
    }
}

impl ViewLayoutLaunchArg for PlainLayout {
    type RuntimeArg<R: Runtime> = ();
    type CompilationArg = ();

    fn register<R: Runtime, B: MemoryArg>(
        _: Self::RuntimeArg<R>,
        buffer: &B,
        ty: Type,
        launcher: &mut KernelLauncher<R>,
    ) {
        <usize as LaunchArg>::register(buffer.len() / ty.vector_size(), launcher);
    }

    fn expand(
        _: &Self::CompilationArg,
        _: Type,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let len = <usize as LaunchArg>::expand(&(), builder);
        PlainLayout::__expand_new(&builder.scope, len)
    }
}

#[cube]
impl Layout for PlainLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        pos
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
