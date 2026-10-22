//! In-kernel evaluation of a [`Compaction`]'s steps ([`CompactionStep`]).

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// A [`Layout`] scaling a physical coordinate by one step per axis: `src[pa] = pos[pa] * step`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct CompactionStep {
    /// The compacted extents this steps through, innermost a line count.
    shape: Coords<u32>,
    #[cube(comptime)]
    steps: Vec<usize>,
}

#[cube]
impl CompactionStep {
    pub(crate) fn new(shape: Coords<u32>, #[comptime] steps: Vec<usize>) -> Self {
        let rank = shape.len();
        comptime!(assert!(
            rank == steps.len(),
            "CompactionStep: shape has {rank} entries but {} steps were given",
            steps.len()
        ));
        CompactionStep { shape, steps }
    }
}

#[cube]
impl Layout for CompactionStep {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.steps.len()) {
            out.push(pos[pa].times(comptime!(self.steps[pa] as u32)));
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.to_dyn()
    }

    /// The compacted box.
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.shape.within(pos)
    }
}
