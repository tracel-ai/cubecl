//! The tile of a level spanning the launch that a walk's plane or unit takes.

use cubecl::prelude::*;

use crate::{
    ComputeScope, Coords, CubeAxis, CubeOrder, Integer, IntegerExpand, Level, Space, cube_positions,
};

/// This plane's or unit's tile of `level`, a level spanning the launch, along each axis of
/// `space`: `(positions, counts)`: along each of the level's axes the tile's index and `1`, an
/// index past the last for an instance past the last tile; along each other axis `0` and every
/// tile. `grid` is the tiles along each axis.
#[cube]
pub(crate) fn launch_tiles(
    #[comptime] space: Space,
    #[comptime] level: Level,
    grid: &Coords<usize>,
) -> (Coords<usize>, Coords<usize>) {
    let order = comptime!(
        level
            .launch_order()
            .expect("a level spanning the launch states the order its instances take")
    );
    // An instance's place in its cube: a plane's row, or a unit's place among all of its units.
    let in_cube = match comptime!(level.coverage().scope()) {
        Some(ComputeScope::Unit) => UNIT_POS as usize,
        _ => UNIT_POS_Y as usize,
    };
    let rank = comptime!(space.rank());
    // The level's axes in the order it names them: the first rides fastest across the blocks, as
    // a cube level's `X` does.
    let named = comptime!(
        level
            .axes()
            .into_iter()
            .map(|axis| space.position(axis))
            .collect::<Vec<_>>()
    );
    // The grid's `x` and `y`, as `Partitioning::cube_count` lays the cubes out:
    // its `z` is always one.
    let cube = CubeAxis::position(CubeAxis::X)
        .plus(CubeAxis::position(CubeAxis::Y).times(CUBE_COUNT_X as usize));
    let (block, local) = if comptime!(order.one_block_a_cube()) {
        (cube, in_cube)
    } else {
        let flat = cube
            .times(comptime!(order.planes_per_cube()).runtime())
            .plus(in_cube);
        let planes = comptime!(order.block_planes()).runtime();
        (flat.divided_by(planes), flat.remainder(planes))
    };

    // Blocks along each named axis, the last one ragged.
    let mut blocks = Coords::<usize>::new();
    #[unroll]
    for p in comptime!(named.clone()) {
        let along = comptime!(order.block_along(space.axis_at(p)));
        blocks.push(
            grid.at(p)
                .plus(comptime!(along - 1).runtime())
                .divided_by(along.runtime()),
        );
    }
    let at = block_coords(
        block,
        &blocks,
        comptime!(named.len()),
        comptime!(order.block_order()),
    );

    let mut positions = Coords::<usize>::new();
    let mut counts = Coords::<usize>::new();
    #[unroll]
    for p in 0..rank {
        match comptime!(named.iter().position(|&q| q == p)) {
            None => {
                positions.push(0usize);
                counts.push(grid.at(p));
            }
            Some(i) => {
                let along = comptime!(order.block_along(space.axis_at(p)));
                // The last named axis rides fastest within a block, as a plane level's does.
                let inner = comptime!(
                    named[i + 1..]
                        .iter()
                        .map(|&q| order.block_along(space.axis_at(q)))
                        .product::<usize>()
                );
                let within = local.divided_by(inner.runtime()).remainder(along.runtime());
                let tile = at.at(i).times(along.runtime()).plus(within);
                positions.push(tile);
                counts.push(1usize);
            }
        }
    }
    (positions, counts)
}

/// Block `block`'s coordinates over `blocks`, `named` axes of them: the first two in `order`, the
/// others outermost, first fastest, the last never wrapped, so a block past the grid lands off it.
#[cube]
fn block_coords(
    block: usize,
    blocks: &Coords<usize>,
    #[comptime] named: usize,
    #[comptime] order: CubeOrder,
) -> Coords<usize> {
    let mut out = Coords::<usize>::new();
    if comptime!(named == 1) {
        out.push(block);
    } else {
        let plane = blocks.at(0).times(blocks.at(1));
        let first_two = if comptime!(named == 2) {
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
            if comptime!(i + 1 == named) {
                out.push(rest);
            } else {
                out.push(rest.remainder(blocks.at(i)));
                rest = rest.divided_by(blocks.at(i));
            }
        }
    }
    out
}
