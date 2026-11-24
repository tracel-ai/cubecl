//! One level's cut of one space ([`Step`]).

use super::{Level, Space};
use crate::{Coords, Dimension, Integer, IntegerExpand};
use cubecl::prelude::*;

/// One level's cut of one space: the tile coordinates a loop handed out, and the space.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Step {
    coords: Coords<u32>,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) level: Level,
    /// Where `level` sits in the nest, outermost `0`.
    #[cube(comptime)]
    pub(crate) depth: usize,
}

#[cube]
impl Step {
    pub(crate) fn new(
        coords: Coords<u32>,
        #[comptime] space: Space,
        #[comptime] level: Level,
        #[comptime] depth: usize,
    ) -> Step {
        Step {
            coords,
            space,
            level,
            depth,
        }
    }

    /// The coordinate along `dimension`; `0` when the dimension is absent.
    pub(crate) fn coord(&self, #[comptime] dimension: Dimension) -> usize {
        if comptime!(self.space.contains(dimension)) {
            self.coords
                .at(comptime!(self.space.position(dimension)))
                .cast::<usize>()
        } else {
            0usize.runtime()
        }
    }
}
