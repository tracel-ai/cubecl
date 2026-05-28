use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use crate::tensor::{
    View,
    launch::{MemoryArg, ViewArg, ViewLayoutLaunchArg},
    layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

/// Tiling as a composable layout: splits each tiled axis into `levels + 1`
/// factors — one grid plus `levels` nested tile sizes — via a mixed-radix
/// decomposition, expanding logical rank R to physical rank `R + levels * n`. The
/// factors are read from the physical shape, laid out `[pre, grid…, level1…, …,
/// levelL…, post]` (coarsest grid first, finest tile last). `levels == 1` is a
/// single split `c -> (c / T, c % T)`.
#[derive(CubeType, Clone)]
pub struct TiledLayout {
    physical_shape: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    #[cube(comptime)]
    num_tiled: usize,
    #[cube(comptime)]
    levels: usize,
}

#[cube]
impl TiledLayout {
    pub fn new(
        physical_shape: CoordsDyn,
        #[comptime] start_axis: usize,
        #[comptime] num_tiled: usize,
        #[comptime] levels: usize,
    ) -> TiledLayout {
        TiledLayout {
            physical_shape,
            start_axis,
            num_tiled,
            levels,
        }
    }
}

#[cube]
impl Layout for TiledLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    /// Logical coords to physical tile coords, laid out
    /// `[pre, grid…, level1…, …, levelL…, post]`. Each tiled axis is decomposed
    /// mixed-radix: the finest level (last block) is the least significant digit.
    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        #[comptime]
        let n = self.num_tiled;
        #[comptime]
        let levels = self.levels;
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut physical = CoordsDyn::new();
        #[unroll]
        for i in 0..self.start_axis {
            physical.push(pos[i]);
        }
        #[unroll]
        for k in 0..comptime!(levels + 1) {
            #[unroll]
            for i in 0..n {
                // Strip the finer blocks, then take this block's digit. The grid
                // (k == 0) keeps the full quotient — it has no enclosing tile.
                let mut divisor = 1u32;
                #[unroll]
                for finer in 0..comptime!(levels + 1) {
                    if comptime!(finer > k) {
                        divisor *= self.physical_shape[comptime!(self.start_axis + finer * n + i)];
                    }
                }
                let digit = pos[comptime!(self.start_axis + i)] / divisor;
                if comptime!(k == 0) {
                    physical.push(digit);
                } else {
                    physical.push(digit % self.physical_shape[comptime!(self.start_axis + k * n + i)]);
                }
            }
        }
        #[unroll]
        for i in 0..comptime!(physical_rank - (self.start_axis + (levels + 1) * n)) {
            physical.push(pos[comptime!(self.start_axis + n + i)]);
        }
        physical
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    /// Logical shape: each tiled axis collapses its `levels + 1` factors back to
    /// their product.
    fn shape(&self) -> Self::Coordinates {
        #[comptime]
        let n = self.num_tiled;
        #[comptime]
        let levels = self.levels;
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut semantic = CoordsDyn::new();
        #[unroll]
        for i in 0..self.start_axis {
            semantic.push(self.physical_shape[i]);
        }
        #[unroll]
        for i in 0..n {
            let mut extent = 1u32;
            #[unroll]
            for k in 0..comptime!(levels + 1) {
                extent *= self.physical_shape[comptime!(self.start_axis + k * n + i)];
            }
            semantic.push(extent);
        }
        #[unroll]
        for i in 0..comptime!(physical_rank - (self.start_axis + (levels + 1) * n)) {
            semantic.push(self.physical_shape[comptime!(self.start_axis + (levels + 1) * n + i)]);
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
    #[cube(comptime)]
    levels: usize,
}

#[cube]
impl Layout for TiledViewLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    /// Logical coords to buffer offset: split to physical coords, then strides.
    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let split = TiledLayout::new(
            self.physical_shape.clone(),
            self.start_axis,
            self.num_tiled,
            self.levels,
        );
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
        let split = TiledLayout::new(
            self.physical_shape.clone(),
            self.start_axis,
            self.num_tiled,
            self.levels,
        );
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

/// Tiling spec for [`tiled_view`]: where the tiled axes start, how many there are,
/// and how many nested tile levels each carries (`levels == 1` is a single split).
/// The per-level sizes are read from the buffer's `[grid…, level1…, …]` dims.
#[derive(Clone, Debug)]
pub struct TileSpec {
    pub start_axis: u8,
    pub num_tiled: usize,
    pub levels: usize,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct TiledViewLayoutCompilationArg {
    physical_shape: <CoordsDyn as LaunchArg>::CompilationArg,
    physical_strides: <CoordsDyn as LaunchArg>::CompilationArg,
    start_axis: u8,
    num_tiled: usize,
    levels: usize,
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
            levels: spec.levels,
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
            levels: arg.levels,
        }
    }
}

/// View type alias for a tiled buffer seen through its logical coordinates.
pub type TiledView<'a, E> = View<'a, E, CoordsDyn>;
pub type TiledViewLaunch<R> = ViewArg<CoordsDyn, R>;
