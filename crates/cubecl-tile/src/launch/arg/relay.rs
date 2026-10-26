//! A destination the cubes of a split contraction write in turns: the [`Relay`].

use cubecl::prelude::*;

use crate::launch::arg::adding::adding_tile;
use crate::launch::arg::tensor::TileArgExpand;
use crate::space::partition::GridCount;
use crate::*;

/// One output box, relayed by the cubes a [`Levels::across`] split hands it: each contracts its
/// own run, then takes its turn at the box's carry in the order of the runs. The first turn writes
/// its partial, each after adds its own into what is there a line at a time, and the last hands the
/// sum on to the output. The carry is the [`TileArg`] the relay was opened on; the turns are passed
/// on through `turns`, one counter a cube ([`Partitioning::relay_counters`]).
///
/// A cube opens its accumulator on the relayed [`tile`](Relay::tile), takes its turn with
/// [`take`](Relay::take), drains into the tile, and passes it on with [`pass`](Relay::pass), every
/// unit of the cube reaching both. A sum whose partials must be rescaled against one another before
/// they add, as an online softmax's are, reads and rescales what the turns before it left
/// ([`carried`](Relay::carried)) in its turn, then drains as any other. The relay publishes a
/// turn's lines at device scope before passing the box on and acquires them after taking it, so it
/// runs only where the runtime hands one cube's writes to another within a dispatch
/// (`device_memory_scope`), and reads its turn with a `u32` atomic add, which every GPU runtime
/// offers. A waiting cube holds its place on the device: the relay rests on a box's earlier runs
/// having started by the time a later one waits, which a grid whose runs are neighbours gives.
#[derive(CubeType)]
pub struct Relay<'a> {
    /// The carry, read by the last turn and by a turn that rescales it.
    carry: Tile,
    /// The carry written in turns: replaced by the first, added into by the rest.
    sink: Tile,
    turns: &'a [Atomic<u32>],
    /// The cube level's walk, whose first region is this cube's box.
    walk: Walk,
    /// This box's counter.
    counter: usize,
    /// This cube's turn: its run's position along the split axis.
    turn: usize,
    /// The turns whose run holds a tile: the leading ones.
    holders: usize,
}

#[cube]
impl<'a> TileArg<'a> {
    /// This operand as the carry of a [`Relay`]: the box the cubes splitting the contraction
    /// along its [`across`](Levels::across) axis take their turns at, under `partitioning`, whose
    /// first level is the cube level. `turns` arrives holding zero and is left holding zero.
    pub fn relay<'t, V: Size>(
        &self,
        partitioning: &Partitioning,
        turns: &'t [Atomic<u32>],
    ) -> Relay<'t> {
        let space = comptime!(partitioning.clone());
        let split = comptime!(Relay::split(&space, &self.spec));
        let walk = partitioning.walk();
        let counter = Relay::counter(partitioning, &walk, split);
        let turn = walk.position_at(split);
        let holders = Relay::holders(partitioning, split);

        let carry = self.tile(partitioning);
        let define!(Served) = self.element;
        let sink = adding_tile::<Served, V>(self, partitioning, turn == 0usize);

        Relay::<'t> {
            carry,
            sink,
            turns,
            walk,
            counter,
            turn,
            holders,
        }
    }
}

#[cube]
impl<'a> Relay<'a> {
    /// Whether this cube's run holds a tile, and so takes a turn: a cube past the last such run
    /// holds nothing and never meets the box. Uniform across the cube.
    pub fn has_turn(&self) -> bool {
        self.turn < self.holders
    }

    /// This cube's box of the carry, relayed: what its accumulator opens on, starting from the
    /// monoid's identity, and what it drains into once it has [`take`](Relay::take)n its turn. The
    /// first turn replaces what the box held; every later one adds into it.
    pub fn tile(&self) -> Tile {
        self.sink.at(&self.walk.region(0usize))
    }

    /// This cube's box of the carry as the turns before this one left it, to read and to rescale
    /// in place once this cube has [`take`](Relay::take)n its turn and before it drains. It holds
    /// nothing on the [first](Relay::is_first) turn, whose drain replaces it.
    pub fn carried(&self) -> Tile {
        self.carry.at(&self.walk.region(0usize))
    }

    /// Whether this cube takes its box's first turn, whose drain replaces what the carry held.
    /// Uniform across the cube.
    pub fn is_first(&self) -> bool {
        self.turn == 0
    }

    /// Whether this cube takes its box's last turn, whose carry [`pass`](Relay::pass) hands on to
    /// the output. Uniform across the cube.
    pub fn is_last(&self) -> bool {
        self.turn + 1 == self.holders
    }

    /// Wait for this cube's turn at its box, and acquire what the turns before it wrote. Every
    /// unit of the cube reaches it.
    ///
    /// The counter is read with `fetch_add(0)`. On Metal, an atomic load spinning on the counter
    /// can keep returning the value it first read long after the previous turn's store has
    /// landed, and the cube then never leaves the loop. A plain `load` serves once the Metal
    /// backend's atomic load reads the counter coherently on every spin.
    pub fn take(&self) {
        if UNIT_POS == 0 {
            loop {
                if self.turns[self.counter].fetch_add(0u32) == self.turn as u32 {
                    break;
                }
            }
        }
        sync_storage();
    }

    /// Hand the box on: to the next turn, or, on the last, into `out` with a cast, once every
    /// plane's lines of the carry are in. Every unit of the cube reaches it.
    pub fn pass(&self, out: &mut Tile) {
        let last = self.is_last();
        // Publishes this turn's lines before the next turn, or the last's copy, reads them.
        sync_storage();
        if last {
            let region = self.walk.region(0usize);
            let mut out_box = out.at(&region);
            out_box.copy_cast_from(&self.carry.at(&region));
        }
        if UNIT_POS == 0 {
            // The last turn leaves the counter at zero, ready for the next launch.
            if last {
                self.turns[self.counter].store(0u32);
            } else {
                self.turns[self.counter].store(self.turn as u32 + 1);
            }
        }
    }

    /// The counter of this cube's box: the cube's position over every axis the cube level
    /// distributes but the `split` one, as one mixed-radix index, the split's coordinate left at
    /// zero.
    fn counter(partitioning: &Partitioning, walk: &Walk, #[comptime] split: usize) -> usize {
        let space = comptime!(partitioning.clone());
        let level = comptime!(space.levels()[0].clone());
        let mut counter = 0usize;
        #[unroll]
        for p in 0..comptime!(space.space().rank()) {
            let axis = comptime!(space.space().axis_at(p));
            if comptime!(level.distributes(axis)) {
                let radix = Self::radix(partitioning, comptime!(level.clone()), p);
                let position = match comptime!(p == split) {
                    true => 0usize.runtime(),
                    false => walk.position_at(p),
                };
                counter = counter.times(radix).plus(position);
            }
        }
        counter
    }

    /// The turns along `split` whose run holds a tile: the leading ones, a run being
    /// `ceil(grid / workers)` tiles.
    fn holders(partitioning: &Partitioning, #[comptime] split: usize) -> usize {
        let space = comptime!(partitioning.clone());
        let level = comptime!(space.levels()[0].clone());
        let axis = comptime!(space.space().axis_at(split));
        let workers = comptime!(match level.count(axis) {
            Some(Count::AllAcross(workers)) => workers,
            _ => unreachable!("Relay::split names an axis spread across cubes"),
        });
        let grid = Self::grid(partitioning, comptime!(level.clone()), split);
        match comptime!(level.cut(axis).spread) {
            Spread::Contiguous => {
                let run = grid
                    .plus(comptime!(workers - 1).runtime())
                    .divided_by(workers.runtime())
                    .max_with(1usize.runtime());
                grid.plus(run.minus(1usize.runtime())).divided_by(run)
            }
            Spread::Interleaved => grid.min_with(workers.runtime()),
        }
    }

    /// The instances the cube level distributes axis `p` to: its workers where spread across them,
    /// its tile count otherwise.
    fn radix(partitioning: &Partitioning, #[comptime] level: Level, #[comptime] p: usize) -> usize {
        let axis = comptime!(partitioning.space().axis_at(p));
        match comptime!(level.count(axis)) {
            Some(Count::AllAcross(workers)) => workers.runtime(),
            _ => Self::grid(partitioning, level, p),
        }
    }

    /// The tiles the cube level cuts axis `p` into.
    fn grid(partitioning: &Partitioning, #[comptime] level: Level, #[comptime] p: usize) -> usize {
        let axis = comptime!(partitioning.space().axis_at(p));
        match comptime!(level.grid(axis)) {
            GridCount::Const(n) => n.runtime(),
            GridCount::Extent(tile) => partitioning.space.count(p, tile),
        }
    }
}

impl Relay<'_> {
    /// The position in `space` of the one axis a relay splits: spread across cubes by the first
    /// level, the cube level, and not spanned by the carry. Refuses any other partitioning.
    fn split(space: &Partitioning, spec: &TileSpec) -> usize {
        let level = &space.levels()[0];
        assert!(
            level.coverage() == Coverage::Distribute(ComputeScope::Cube)
                && level.shared_by().is_none(),
            "TileArg::relay: the first level must hand each cube its own boxes; a relay's turns are \
             taken by the cubes of one box"
        );
        let split: Vec<usize> = (0..space.space().rank())
            .filter(|&p| {
                let axis = space.space().axis_at(p);
                matches!(level.count(axis), Some(Count::AllAcross(_)))
                    && !spec.axes().contains(&axis)
            })
            .collect();
        assert!(
            split.len() == 1,
            "TileArg::relay: the carry must leave out exactly one axis the cube level spreads across \
             cubes (`Levels::across`), the one whose runs take the turns; it leaves out {}",
            split.len()
        );
        split[0]
    }
}

impl Partitioning {
    /// The turn counters a [`Relay`] under this partitioning passes its boxes on through: one
    /// `u32` a cube, holding zero.
    pub fn relay_counters(&self) -> usize {
        self.instances(Coverage::Distribute(ComputeScope::Cube)) as usize
    }
}
