//! Where a contraction runs, read off the tile it is called on ([`Descent`]), and the run there
//! ([`Descent::contract`]).

use cubecl::prelude::*;

use super::leaf::{Side, deep_load, memory};
use crate::tile::base::witnessed_space;
use crate::*;

/// Where `acc += lhs · rhs` runs, from the accumulator it is called on and the level below it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Descent {
    /// `acc` holds one fragment or block, or is a memory window: its instruction runs here.
    Here,
    /// `acc` is a grid of fragments and the level below steps the contraction: each of its regions
    /// contracts in turn, through the grid's instruction.
    Steps(Instruction),
    /// `acc` is a grid of fragments and the level below only moves its cells: each operand's
    /// fragments load once in the instruction's form, then every cell contracts.
    Cells(Instruction),
}

impl Descent {
    /// The descent into an accumulator placed at `acc`, a grid contracting through `grid`'s
    /// instruction or `None` where it holds one fragment, from operands spanning `lhs` and `rhs`.
    pub(crate) fn of(grid: Option<Instruction>, acc: &Placement, lhs: &Space, rhs: &Space) -> Self {
        let Some(instruction) = grid else {
            return Descent::Here;
        };
        let below = acc.below().first().unwrap_or_else(|| {
            panic!("mma: a grid of fragments contracts at a level with its cells below it")
        });
        let operands = Space::merge(&[lhs, rhs]);
        let steps_contraction = below
            .axes()
            .iter()
            .any(|&axis| operands.contains(axis) && !acc.space.contains(axis));
        match steps_contraction {
            true => Descent::Steps(instruction),
            false => Descent::Cells(instruction),
        }
    }
}

#[cube]
impl Descent {
    /// The descent a contraction into `acc` takes.
    pub(crate) fn of_tile(acc: &Tile, lhs: &Tile, rhs: &Tile) -> comptime_type!(Descent) {
        match &acc.kind {
            TileKind::PlanePartition(p) => {
                let grid = p.grid();
                comptime!(Descent::of(
                    grid,
                    &acc.place,
                    &lhs.place.space,
                    &rhs.place.space
                ))
            }
            TileKind::PlaneTile(_)
            | TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(Descent::Here),
        }
    }

    /// The leaf contraction `acc += lhs · rhs`, dispatched on the accumulator's form.
    pub(crate) fn contract<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
        acc: &mut Tile,
        lhs: &Tile,
        rhs: &Tile,
    ) {
        // A clone holds the same fragments: writes through it land in `acc`.
        let grid = acc.clone();
        let descent = Self::of_tile(&grid, lhs, rhs);
        match comptime!(descent) {
            Descent::Here => Self::here::<E, Lhs, Rhs>(acc, lhs, rhs),
            Descent::Steps(instruction) => Self::steps::<E, Lhs, Rhs>(&grid, lhs, rhs, instruction),
            Descent::Cells(instruction) => Self::cells::<E, Lhs, Rhs>(&grid, lhs, rhs, instruction),
        }
    }

    /// [`Descent::Steps`]: each region of the level below, walked over the whole contraction's box
    /// (`acc`'s alone lacks the contracted axes).
    ///
    /// A factor the tensor cores read through a landing, scaled or packed, lands once for every
    /// step, a whole load at a time: a step of the instruction may be narrower than one of its
    /// loads, and its scales are read once for the box rather than once a step.
    fn steps<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
        acc: &Tile,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] instruction: Instruction,
    ) {
        let out = comptime!(acc.place.space.clone());
        let lhs = lhs.landed_for::<Lhs>(comptime!(Side::Lhs), comptime!(out.clone()), instruction);
        let rhs = rhs.landed_for::<Rhs>(comptime!(Side::Rhs), out, instruction);
        let operands = comptime!(Space::merge(&[
            &acc.place.space,
            &lhs.place.space,
            &rhs.place.space
        ]));
        let space = witnessed_space(operands, acc, &lhs, &rhs);
        let walk = Region::rooted(
            &space,
            comptime!(acc.place.levels.clone()),
            comptime!(acc.place.depth),
        )
        .walk();
        for region in walk.unrolled() {
            let mut acc_region = acc.at(&region);
            Self::contract::<E, Lhs, Rhs>(&mut acc_region, &lhs.at(&region), &rhs.at(&region));
        }
    }

    /// [`Descent::Cells`]: each operand's fragments loaded once, then every cell of the grid.
    fn cells<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
        acc: &Tile,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] instruction: Instruction,
    ) {
        let load = deep_load(lhs, rhs);
        let lhs = PlanePartition::operand::<Lhs>(lhs, acc, instruction, load);
        let rhs = PlanePartition::operand::<Rhs>(rhs, acc, instruction, load);
        for cell in acc.walk().unrolled() {
            let mut acc_cell = acc.at(&cell);
            Self::contract::<E, Lhs, Rhs>(&mut acc_cell, &lhs.at(&cell), &rhs.at(&cell));
        }
    }

    /// [`Descent::Here`]: `acc += lhs · rhs` into the one fragment or block `acc` holds, or the
    /// memory window it is.
    fn here<E: Numeric, Lhs: Numeric, Rhs: Numeric>(acc: &mut Tile, lhs: &Tile, rhs: &Tile) {
        let space = comptime!(acc.place.space.clone());
        let tile_kind = &mut acc.kind;
        match tile_kind {
            TileKind::PlaneTile(t) => t.mma::<E, Lhs, Rhs>(lhs, rhs, space),
            TileKind::PlanePartition(p) => {
                let mut t = p.at(0usize, 0usize);
                t.mma::<E, Lhs, Rhs>(lhs, rhs, space)
            }
            TileKind::Memory(g) => {
                let contraction = g.stated_contraction();
                memory::contract::<E, Lhs, Rhs>(
                    g,
                    lhs,
                    rhs,
                    space,
                    contraction.block,
                    contraction.semiring,
                )
            }
            TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("mma: a procedural tile and the plane's units are not an accumulator sink")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A 2×2 grid of 8×8 fragments, with `below` the level under it.
    fn grid_over(below: Level) -> Placement {
        Placement::new(Space::new(&[(M, 16), (N, 16)]), 0, vec![below])
    }

    fn descent(grid: Option<Instruction>, below: Level) -> Descent {
        let lhs = Space::new(&[(M, 16), (K, 16)]);
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        Descent::of(grid, &grid_over(below), &lhs, &rhs)
    }

    /// One fragment runs its instruction where it is called, whatever lies below.
    #[test]
    fn one_fragment_contracts_here() {
        let below = Levels::leaf(&[(M, 8), (N, 8)])
            .walk(&[(M, 2), (N, 2)])
            .level();
        assert_eq!(descent(None, below), Descent::Here);
    }

    /// A level that steps only the grid's cells loads each operand's fragments once above it.
    #[test]
    fn a_grid_over_its_cells_loads_its_operands_once() {
        let below = Levels::leaf(&[(M, 8), (N, 8)])
            .walk(&[(M, 2), (N, 2)])
            .level();
        assert_eq!(
            descent(Some(Instruction::Cmma), below),
            Descent::Cells(Instruction::Cmma)
        );
    }

    /// A level that also steps `K` contracts each of its regions in turn, `K` included.
    #[test]
    fn a_grid_over_the_contraction_steps_it() {
        let below = Levels::leaf(&[(M, 8), (N, 8), (K, 8)])
            .walk(&[(M, 2), (N, 2), (K, 2)])
            .level();
        assert_eq!(
            descent(Some(Instruction::Cmma), below),
            Descent::Steps(Instruction::Cmma)
        );
    }
}
