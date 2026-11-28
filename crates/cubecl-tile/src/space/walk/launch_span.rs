//! The tile of a level spanning the launch that a walk's plane or unit takes.

use cubecl::prelude::*;

use crate::{
    ComputeScope, Coords, CubeAxis, CubeOrder, Integer, IntegerExpand, Level, Space, cube_positions,
};

/// This plane's or unit's tiles of `level`, a level spanning the launch, along each dimension of
/// `space`: `(positions, counts, scales)`: along each of the level's dimensions the first tile's
/// index, how many it walks and how far apart they lie: one tile, an index past the last for an
/// instance past the last tile; under a [capped](crate::space::PlaneOrder::capped) order, every
/// tile a whole launch of planes apart until the last. Along each other dimension `0`, every tile,
/// `1`. `grid` is the tiles along each dimension.
#[cube]
pub(crate) fn launch_tiles(
    #[comptime] space: &Space,
    #[comptime] level: &Level,
    grid: &Coords<usize>,
) -> (Coords<usize>, Coords<usize>, Coords<usize>) {
    let order = level
        .launch_order()
        .expect("a level spanning the launch should state the order its instances take");
    // An instance's place in its cube: a plane's row, or a unit's place among all of its units.
    let in_cube = match level.coverage().scope() {
        Some(ComputeScope::Unit) => UNIT_POS as usize,
        _ => UNIT_POS_Y as usize,
    };
    let rank = space.rank();
    // The level's dimensions in the order it names them: the first rides fastest across the blocks,
    // as a cube level's `X` does.
    let named = comptime!(
        level
            .dimensions()
            .into_iter()
            .map(|dimension| space.position(dimension))
            .collect::<Vec<_>>()
    );
    // The grid's `x` and `y`, as `Partitioning::cube_count` lays the cubes out:
    // its `z` is always one.
    let cube = CubeAxis::position(CubeAxis::X)
        .plus(CubeAxis::position(CubeAxis::Y).times(CUBE_COUNT_X as usize));
    let (block, local) = if order.one_block_a_cube() {
        (cube, in_cube)
    } else {
        let flat = cube.times(order.planes_per_cube().runtime()).plus(in_cube);
        let planes = order.block_planes().runtime();
        (flat.divided_by(planes), flat.remainder(planes))
    };

    // Blocks along each named dimension, the last one ragged.
    let mut blocks = Coords::<usize>::new();
    #[unroll]
    for p in named.clone() {
        let along = order.block_along(space.dimension_at(p));
        blocks.push(
            grid.at(p)
                .plus((along - 1).runtime())
                .divided_by(along.runtime()),
        );
    }
    let at = block_coords(block, &blocks, named.len(), order.block_order());

    let capped = order.cap().is_some();
    let mut positions = Coords::<usize>::new();
    let mut counts = Coords::<usize>::new();
    let mut scales = Coords::<usize>::new();
    #[unroll]
    for p in 0..rank {
        match comptime!(named.iter().position(|&q| q == p)) {
            None => {
                positions.push(0usize);
                counts.push(grid.at(p));
                scales.push(1usize);
            }
            Some(i) => {
                let along = order.block_along(space.dimension_at(p));
                // The last named dimension rides fastest within a block, as a plane level's does.
                let inner = comptime!(
                    named[i + 1..]
                        .iter()
                        .map(|&q| order.block_along(space.dimension_at(q)))
                        .product::<usize>()
                );
                let within = local.divided_by(inner.runtime()).remainder(along.runtime());
                let tile = at.at(i).times(along.runtime()).plus(within);
                positions.push(tile);
                if capped {
                    // A launch's planes apart: every cube's block, a tile a plane.
                    let stride = CUBE_COUNT.times(along.runtime());
                    let first = tile.min_with(grid.at(p));
                    counts.push(
                        grid.at(p)
                            .minus(first)
                            .plus(stride)
                            .minus(1usize.runtime())
                            .divided_by(stride),
                    );
                    scales.push(stride);
                } else {
                    counts.push(1usize);
                    scales.push(1usize);
                }
            }
        }
    }
    (positions, counts, scales)
}

/// Block `block`'s coordinates over `blocks`, `named` dimensions of them: the first two in `order`,
/// the others outermost, first fastest, the last never wrapped, so a block past the grid lands off
/// it.
#[cube]
fn block_coords(
    block: usize,
    blocks: &Coords<usize>,
    #[comptime] named: usize,
    #[comptime] order: CubeOrder,
) -> Coords<usize> {
    let mut out = Coords::<usize>::new();
    if named == 1 {
        out.push(block);
    } else {
        let plane = blocks.at(0).times(blocks.at(1));
        let first_two = if named == 2 {
            block
        } else {
            block.remainder(plane)
        };
        let (x, y) = cube_positions(first_two, (blocks.at(0), blocks.at(1)), order);
        out.push(x);
        out.push(y);
        let mut rest = block.divided_by(plane);
        #[unroll]
        for i in 2..named {
            if i + 1 == named {
                out.push(rest);
            } else {
                out.push(rest.remainder(blocks.at(i)));
                rest = rest.divided_by(blocks.at(i));
            }
        }
    }
    out
}
