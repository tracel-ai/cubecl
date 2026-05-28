use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use crate::tensor::{
    View,
    launch::{MemoryArg, ViewArg, ViewLayoutLaunchArg},
    layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

/// Tiling as a composable layout: splits each tiled axis `c -> (c / T, c % T)`,
/// expanding logical rank R to physical rank R + n. Tile sizes `T` come from the
/// physical shape's tile dims.
#[derive(CubeType, Clone)]
pub struct TiledLayout {
    physical_shape: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    #[cube(comptime)]
    num_tiled: usize,
}

#[cube]
impl TiledLayout {
    pub fn new(
        physical_shape: CoordsDyn,
        #[comptime] start_axis: usize,
        #[comptime] num_tiled: usize,
    ) -> TiledLayout {
        TiledLayout {
            physical_shape,
            start_axis,
            num_tiled,
        }
    }
}

#[cube]
impl Layout for TiledLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    /// Logical coords to physical tile coords, laid out `[pre, grid, tile, post]`.
    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        #[comptime]
        let n = self.num_tiled;
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut physical = CoordsDyn::new();
        #[unroll]
        for i in 0..self.start_axis {
            physical.push(pos[i]);
        }
        #[unroll]
        for i in 0..n {
            let tile = self.physical_shape[comptime!(self.start_axis + n + i)];
            physical.push(pos[comptime!(self.start_axis + i)] / tile);
        }
        #[unroll]
        for i in 0..n {
            let tile = self.physical_shape[comptime!(self.start_axis + n + i)];
            physical.push(pos[comptime!(self.start_axis + i)] % tile);
        }
        #[unroll]
        for i in 0..comptime!(physical_rank - (self.start_axis + 2 * n)) {
            physical.push(pos[comptime!(self.start_axis + n + i)]);
        }
        physical
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    /// Logical shape: each tiled axis collapses its (grid, tile) pair to grid * tile.
    fn shape(&self) -> Self::Coordinates {
        #[comptime]
        let n = self.num_tiled;
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut semantic = CoordsDyn::new();
        #[unroll]
        for i in 0..self.start_axis {
            semantic.push(self.physical_shape[i]);
        }
        #[unroll]
        for i in 0..n {
            let grid = self.physical_shape[comptime!(self.start_axis + i)];
            let tile = self.physical_shape[comptime!(self.start_axis + n + i)];
            semantic.push(grid * tile);
        }
        #[unroll]
        for i in 0..comptime!(physical_rank - (self.start_axis + 2 * n)) {
            semantic.push(self.physical_shape[comptime!(self.start_axis + 2 * n + i)]);
        }
        semantic
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut is_valid = true;
        #[unroll]
        for i in 0..bounds.len() {
            is_valid = is_valid && pos[i] < bounds[i];
        }
        is_valid
    }
}

/// Tiling above a strided buffer: [`TiledLayout`]'s split, then per-axis strides
/// to a linear offset. The buffer is a plain strided tensor; tiling is explicit
/// (`start_axis` / `num_tiled`), with no `Tiler` metadata on it.
#[derive(CubeType, Clone)]
pub struct TiledViewLayout {
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    #[cube(comptime)]
    num_tiled: usize,
}

#[cube]
impl Layout for TiledViewLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    /// Logical coords to buffer offset: split to physical coords, then strides.
    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let split = TiledLayout::new(self.physical_shape.clone(), self.start_axis, self.num_tiled);
        let physical = split.to_source_pos(pos);

        let mut offset = 0u32;
        #[unroll]
        for i in 0..self.physical_strides.len() {
            offset += physical[i] * self.physical_strides[i];
        }
        offset as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        let split = TiledLayout::new(self.physical_shape.clone(), self.start_axis, self.num_tiled);
        split.shape()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut is_valid = true;
        #[unroll]
        for i in 0..bounds.len() {
            is_valid = is_valid && pos[i] < bounds[i];
        }
        is_valid
    }
}

/// Tiling spec for [`tiled_view`]: where the tiled axes start and how many there
/// are. The tile sizes are read from the buffer's trailing dims.
#[derive(Clone, Debug)]
pub struct TileSpec {
    pub start_axis: u8,
    pub num_tiled: usize,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct TiledViewLayoutCompilationArg {
    physical_shape: <CoordsDyn as LaunchArg>::CompilationArg,
    physical_strides: <CoordsDyn as LaunchArg>::CompilationArg,
    start_axis: u8,
    num_tiled: usize,
}

impl ViewLayoutLaunchArg for TiledViewLayout {
    type RuntimeArg<R: Runtime> = TileSpec;
    type CompilationArg = TiledViewLayoutCompilationArg;

    fn register<R: Runtime, B: MemoryArg>(
        spec: Self::RuntimeArg<R>,
        buffer: &B,
        _ty: Type,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let shape = buffer.shape();
        let strides = buffer.strides();

        let shape_arg: <CoordsDyn as LaunchArg>::RuntimeArg<R> =
            shape.iter().map(|&s| s as u32).collect();
        let strides_arg: <CoordsDyn as LaunchArg>::RuntimeArg<R> =
            strides.iter().map(|&s| s as u32).collect();

        TiledViewLayoutCompilationArg {
            physical_shape: <CoordsDyn as LaunchArg>::register(shape_arg, launcher),
            physical_strides: <CoordsDyn as LaunchArg>::register(strides_arg, launcher),
            start_axis: spec.start_axis,
            num_tiled: spec.num_tiled,
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        _ty: Type,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        TiledViewLayoutExpand {
            physical_shape: <CoordsDyn as LaunchArg>::expand(&arg.physical_shape, builder),
            physical_strides: <CoordsDyn as LaunchArg>::expand(&arg.physical_strides, builder),
            start_axis: arg.start_axis as usize,
            num_tiled: arg.num_tiled,
        }
    }
}

/// View type alias for a tiled buffer seen through its logical coordinates.
pub type TiledView<E, IO = cubecl_core::prelude::ReadOnly> = View<E, CoordsDyn, IO>;
pub type TiledViewLaunch<R> = ViewArg<CoordsDyn, R>;
