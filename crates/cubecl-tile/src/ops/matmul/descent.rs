//! Where a contraction runs, read off the tile it is called on ([`Descent`]), and the run there
//! ([`Descent::contract`]).

use cubecl::prelude::*;

use super::leaf::{Side, deep_load, memory};
use crate::tile::base::witnessed_space;
use crate::tile::kind::plane::mma::lane_runs::LaneMap;
use crate::*;

/// Where `acc += lhs · rhs` runs, from the accumulator it is called on and the level below it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Descent {
    /// `acc` holds one fragment or block, or is a memory window: its instruction runs here.
    Here,
    /// `acc` is a grid of fragments and the level below steps the contraction, or one cmma
    /// fragment under a level that does: each of its regions contracts in turn, through the
    /// grid's instruction.
    Steps(Instruction),
    /// `acc` is a grid of fragments and the level below only moves its cells: each operand's
    /// fragments load once in the instruction's form, then every cell contracts.
    Cells(Instruction),
}

impl Descent {
    /// The descent into an accumulator placed at `acc`, a grid contracting through `grid`'s
    /// instruction or `None` where it holds one fragment of `fragment`, which loads its factors
    /// deeper than its instruction itself where `loads_deep`, from operands spanning `lhs` and
    /// `rhs`.
    ///
    /// A cmma fragment reads its factors no deeper than its instruction, nor does a manual-mma one
    /// where the target lays out no lane runs to load deeper ones in, so one such fragment under a
    /// level that walks the contraction steps it: run here, it would ask the device for a
    /// fragment as deep as the whole walk.
    pub(crate) fn of(
        grid: Option<Instruction>,
        fragment: Instruction,
        loads_deep: bool,
        acc: &Placement,
        lhs: &Space,
        rhs: &Space,
    ) -> Self {
        let operands = Space::merge(&[lhs, rhs]);
        let walks_contraction = |level: &Level| {
            level
                .axes()
                .iter()
                .any(|&axis| operands.contains(axis) && !acc.space.contains(axis))
        };
        let below = acc.below();
        let Some(instruction) = grid else {
            return match !loads_deep && below.iter().any(walks_contraction) {
                true => Descent::Steps(fragment),
                false => Descent::Here,
            };
        };
        let below = below.first().unwrap_or_else(|| {
            panic!("mma: a grid of fragments contracts at a level with its cells below it")
        });
        match walks_contraction(below) {
            true => Descent::Steps(instruction),
            false => Descent::Cells(instruction),
        }
    }
}

/// Whether one fragment of `fragment` loads factors of `element` deeper than its instruction
/// itself, as the device compiled for lays it out: the crate's transport, in lane runs, for
/// 16-bit factors on a target whose instruction lane runs read.
#[cube]
fn loads_deep(
    #[comptime] fragment: Instruction,
    #[comptime] element: ElemType,
) -> comptime_type!(bool) {
    let target = comptime::target_properties().comptime();
    comptime!(
        matches!(fragment, Instruction::Mma { .. })
            && LaneMap::try_from(&target).is_ok()
            && element.size_bits() == 16
    )
}

/// Whether the level below `acc` steps only axes the sum does not hold: the contraction.
fn steps_contraction_only(acc: &Placement) -> bool {
    acc.below()
        .first()
        .is_some_and(|level| level.axes().iter().all(|&axis| !acc.space.contains(axis)))
}

#[cube]
impl Descent {
    /// The descent a contraction into `acc` takes.
    pub(crate) fn of_tile(acc: &Tile, lhs: &Tile, rhs: &Tile) -> comptime_type!(Descent) {
        match &acc.kind {
            TileKind::PlanePartition(p) => {
                let grid = p.grid();
                let fragment = p.frags.index(0usize).instruction();
                let loads_deep = loads_deep(fragment, lhs.element);
                comptime!(Descent::of(
                    grid,
                    fragment,
                    loads_deep,
                    &acc.place,
                    &lhs.place.space,
                    &rhs.place.space
                ))
            }
            TileKind::PlaneTile(t) => {
                let fragment = t.instruction();
                let loads_deep = loads_deep(fragment, lhs.element);
                comptime!(Descent::of(
                    None,
                    fragment,
                    loads_deep,
                    &acc.place,
                    &lhs.place.space,
                    &rhs.place.space
                ))
            }
            TileKind::Memory(_)
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
        let lhs = lhs.landed_for::<Lhs>(comptime!(Side::Lhs), &out, instruction);
        let rhs = rhs.landed_for::<Rhs>(comptime!(Side::Rhs), &out, instruction);
        let operands = comptime!(Space::merge(&[
            &acc.place.space,
            &lhs.place.space,
            &rhs.place.space
        ]));
        let space = witnessed_space(&operands, acc, &lhs, &rhs);
        let walk = Region::rooted(
            &space,
            comptime!(acc.place.levels.clone()),
            comptime!(acc.place.depth),
        )
        .walk();
        // A level that steps the contraction alone, over operands in memory, only offsets their
        // windows and never picks a fragment of the sum: one rolled loop, so a deep walk is not
        // expanded step by step.
        let contraction_only = comptime!(steps_contraction_only(&acc.place));
        let (lhs_memory, rhs_memory) = (lhs.is_memory(), rhs.is_memory());
        if comptime!(contraction_only && lhs_memory && rhs_memory) {
            for region in walk {
                let mut acc_region = acc.at(&region);
                Self::contract::<E, Lhs, Rhs>(&mut acc_region, &lhs.at(&region), &rhs.at(&region));
            }
        } else {
            for region in walk.unrolled() {
                let mut acc_region = acc.at(&region);
                Self::contract::<E, Lhs, Rhs>(&mut acc_region, &lhs.at(&region), &rhs.at(&region));
            }
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
            TileKind::PlaneTile(t) => t.mma::<E, Lhs, Rhs>(lhs, rhs, &space),
            TileKind::PlanePartition(p) => {
                let mut t = p.at(0usize, 0usize);
                t.mma::<E, Lhs, Rhs>(lhs, rhs, &space)
            }
            TileKind::Memory(g) => {
                let contraction = g.stated_contraction();
                memory::contract::<E, Lhs, Rhs>(
                    g,
                    lhs,
                    rhs,
                    &space,
                    contraction.matmul,
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

    /// The descent into a grid of cmma fragments, or of one cmma fragment where `grid` is `None`.
    fn descent(grid: Option<Instruction>, below: Level) -> Descent {
        descent_of(grid, Instruction::Cmma, false, below)
    }

    fn descent_of(
        grid: Option<Instruction>,
        fragment: Instruction,
        loads_deep: bool,
        below: Level,
    ) -> Descent {
        let lhs = Space::new(&[(M, 16), (K, 16)]);
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        Descent::of(grid, fragment, loads_deep, &grid_over(below), &lhs, &rhs)
    }

    fn mma() -> Instruction {
        Instruction::Mma {
            io: MmaIo::default(),
            operands: MmaOperands::Plain,
        }
    }

    /// One fragment under levels that leave the contraction whole runs its instruction where it
    /// is called.
    #[test]
    fn one_fragment_contracts_here() {
        let below = Levels::leaf(&[(M, 8), (N, 8)])
            .walk(&[(M, 2), (N, 2)])
            .level();
        assert_eq!(descent(None, below), Descent::Here);
    }

    /// One cmma fragment under a level that walks `K` steps it: run here, it would load a fragment
    /// as deep as the whole walk, which a device offering only its instruction's depth (Metal's
    /// `8x8x8`) cannot compile.
    #[test]
    fn one_cmma_fragment_over_the_contraction_steps_it() {
        let below = Levels::leaf(&[(M, 16), (N, 16), (K, 8)])
            .walk(&[(K, 2)])
            .level();
        assert_eq!(descent(None, below), Descent::Steps(Instruction::Cmma));
    }

    /// One manual-mma fragment that loads deeper fragments in lane runs walks a contraction deeper
    /// than its instruction itself, so it still runs where it is called; one that cannot steps it.
    #[test]
    fn one_manual_mma_fragment_over_the_contraction_contracts_here_where_it_loads_deep() {
        let below = || {
            Levels::leaf(&[(M, 16), (N, 16), (K, 8)])
                .walk(&[(K, 2)])
                .level()
        };
        assert_eq!(descent_of(None, mma(), true, below()), Descent::Here);
        assert_eq!(
            descent_of(None, mma(), false, below()),
            Descent::Steps(mma())
        );
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
