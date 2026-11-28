//! The [`Walk`]: the regions one [`Level`] of a [`Space`] hands the instance running the code.

use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::dialect::LevelOp;
use crate::{
    ComputeScope, Coords, Dimension, Extent, Integer, IntegerExpand, Level, Planes, Region,
    RegionExpand, Space, StageOwner,
};

use super::distribution::{DimensionDistribution, Distribution, Workers};
use super::launch_span::launch_tiles;
use super::swizzled::{in_plane_dimensions, swizzled_positions};
use crate::Count;
use crate::Spread;
use crate::space::partition::GridCount;

/// The runtime odometer over a [`Space`]'s tiles under one [`Level`].
#[derive(CubeType)]
pub struct Walk {
    /// The level's tiles along each dimension, which a reversed dimension counts back from.
    tiles: Coords<usize>,
    counts: Coords<usize>,
    positions: Coords<usize>,
    scales: Coords<usize>,
    /// Per-dimension routed coordinate ([`routed`](Walk::routed)), `0` elsewhere.
    route: Coords<u32>,
    /// Which dimensions [`route`](Self::route) speaks for.
    #[cube(comptime)]
    routed_at: Vec<usize>,
    base: usize,
    steps: usize,
    /// Whether the walk takes some of its steps alone ([`range`](Walk::range)).
    #[cube(comptime)]
    ranged: bool,
    pub(crate) parent: Region,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) level: Level,
    /// How the level distributes each dimension of the space, in dimension order.
    #[cube(comptime)]
    distributes: Vec<DimensionDistribution>,
    /// Whether iterating this walk unrolls.
    #[cube(comptime)]
    pub(crate) unroll: bool,
    /// The steps every instance takes and the most any takes, where the host knows both though
    /// not this instance's own: an unrolled walk writes the most out and guards the rest.
    #[cube(comptime)]
    pub(crate) bounds: Option<(usize, usize)>,
    /// The dimension this walk steps a run along ([`Region::walk_within`]), where it steps nothing
    /// else and in order: every region but the last then lies inside the run.
    #[cube(comptime)]
    pub(crate) run: Option<Dimension>,
}

#[cube]
impl Walk {
    // The loops are `#[unroll]`ed cube loops over a comptime `Vec`, not host iteration.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn of(space: &Space, #[comptime] level: Level, parent: Region) -> Walk {
        let host = comptime!(space.clone());
        let rank = host.rank();
        // A swizzled cube level decodes its two in-plane dimensions jointly from the flat dispatch
        // index.
        let in_plane = comptime!(
            level
                .order()
                .swizzles()
                .then(|| in_plane_dimensions(&host, &level))
        );
        let distributes = comptime!(
            (0..rank)
                .map(|p| DimensionDistribution::new(&level, &host, p, in_plane))
                .collect::<Vec<_>>()
        );

        let mut grid = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(level.grid(space.dimension_at(p))) {
                GridCount::Const(n) => grid.push(n.runtime()),
                GridCount::Extent(tile) => grid.push(space.count(p, tile)),
            }
        }

        // A level distributed over a scope reaches the IR as one `tile.level` op.
        let (counts, positions, scales) = match level.coverage().scope() {
            Some(_) => intrinsic!(|scope| LevelOp::trace(scope, |scope| {
                Walk::__expand_tiles_of(scope, &host, &level, distributes.clone(), &grid)
            })),
            None => Walk::tiles_of(&host, &level, distributes.clone(), &grid),
        };

        // Computed whole, not accumulated, so a static walk can unroll.
        let steps = counts.product(comptime!((0..rank).collect::<Vec<_>>()));

        let bounds = comptime!(Walk::bounds(&host, &level));
        Walk {
            tiles: grid,
            counts,
            positions,
            scales,
            route: Coords::constant(comptime!(vec![0; rank])),
            routed_at: comptime!(Vec::new()),
            base: 0usize,
            steps,
            ranged: comptime!(false),
            parent,
            distributes,
            space: host,
            level,
            unroll: comptime!(false),
            bounds,
            run: comptime!(None),
        }
    }

    /// The walk over a copy's `lines` lines that `level` deals ([`Level::copied_by`]): each region
    /// holds one line, at its coordinate along [`Level::COPIED`]. The turns are written out where
    /// each unit takes a known few of them, the last guarded where it can run past the lines; a
    /// stride from the unit's place otherwise.
    pub(crate) fn copy(#[comptime] level: &Level, lines: usize) -> Walk {
        let known = lines.constant();
        let extent = comptime!(match known {
            Some(lines) => Extent::Static(lines as usize),
            None => Extent::Dynamic,
        });
        let mut sizes = Sequence::<usize>::new();
        if comptime!(known.is_none()) {
            sizes.push(lines);
        }
        let space = Space::with_sizes(
            &comptime!(Space::from_extents(&[(Level::COPIED, extent)])),
            sizes,
        );
        let walk = Region::rooted(&space, comptime!(vec![level.clone()]), 0usize).walk();
        let written = comptime!(Walk::copy_turns_written(
            level,
            known.map(|lines| lines as usize)
        ));
        if written { walk.unrolled() } else { walk }
    }

    /// `(counts, positions, scales)` of this instance at `level`.
    fn tiles_of(
        #[comptime] host: &Space,
        #[comptime] level: &Level,
        #[comptime] distributes: Vec<DimensionDistribution>,
        grid: &Coords<usize>,
    ) -> (Coords<usize>, Coords<usize>, Coords<usize>) {
        match level.spans_launch() {
            true => Walk::spanning_launch(host, level, grid),
            false => Walk::dealt(host, level, distributes, grid),
        }
    }

    /// `(counts, positions, scales)` of a level spanning the launch: along each of its dimensions,
    /// the one tile this plane's or unit's place in the launch names, past the last for one past
    /// the last tile, which the binding masks; every tile of the others.
    fn spanning_launch(
        #[comptime] host: &Space,
        #[comptime] level: &Level,
        grid: &Coords<usize>,
    ) -> (Coords<usize>, Coords<usize>, Coords<usize>) {
        let (positions, counts, scales) = launch_tiles(host, level, grid);
        (counts, positions, scales)
    }

    /// `(counts, positions, scales)` of a level dealt to the instances of one cube, or of the
    /// grid's cubes, each dimension as `distributes` says.
    #[allow(clippy::needless_range_loop)]
    fn dealt(
        #[comptime] host: &Space,
        #[comptime] level: &Level,
        #[comptime] distributes: Vec<DimensionDistribution>,
        grid: &Coords<usize>,
    ) -> (Coords<usize>, Coords<usize>, Coords<usize>) {
        let rank = host.rank();
        // Computed whole, so a constant grid's decode stays constant too.
        let mut instances = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(distributes[p].clone()) {
                DimensionDistribution::Distributed(Distribution { workers, .. }) => {
                    match comptime!(workers) {
                        Workers::Across(workers) => instances.push(workers.runtime()),
                        // A count the launch states is compiled in, any other read off it.
                        Workers::Units {
                            within,
                            units,
                            fillers,
                        } => instances
                            .push(DimensionDistribution::units_within(within, units, fillers)),
                        Workers::Grid => instances.push(grid.at(p)),
                    }
                }
                DimensionDistribution::Walked => instances.push(1usize),
            }
        }
        let swizzled = swizzled_positions(host, level, &instances);

        let mut counts = Coords::<usize>::new();
        let mut positions = Coords::<usize>::new();
        let mut scales = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(distributes[p].clone()) {
                DimensionDistribution::Walked => {
                    counts.push(grid.at(p));
                    positions.push(0usize);
                    scales.push(1usize);
                }
                DimensionDistribution::Distributed(Distribution {
                    scope: compute_scope,
                    cube_axis,
                    spread,
                    workers,
                    divides,
                    inner,
                    unspanned,
                    swizzled: joint,
                }) => {
                    // Mixed-radix stride over the later same-dimension dimensions, spanned or not.
                    let inner_weight = instances.product(inner) * comptime!(unspanned).runtime();
                    let position = match comptime!((joint, workers)) {
                        (Some(i), _) => swizzled.at(i),
                        (
                            None,
                            Workers::Units {
                                within, fillers, ..
                            },
                        ) => DimensionDistribution::unit_within(within, fillers),
                        (None, _) => DimensionDistribution::hardware(compute_scope, cube_axis)
                            .divided_by(inner_weight)
                            .remainder(instances.at(p)),
                    };
                    let run = match comptime!(workers) {
                        Workers::Across(workers) => grid
                            .at(p)
                            .plus(comptime!(workers - 1).runtime())
                            .divided_by(workers.runtime()),
                        // Units taking turns, where the host proved the deal even, each take
                        // every `instances`-th tile; elsewhere their count is reckoned per unit.
                        Workers::Units { .. } if divides => grid.at(p).divided_by(instances.at(p)),
                        Workers::Grid | Workers::Units { .. } => 1usize.runtime(),
                    };
                    counts.push(DimensionDistribution::tiles(
                        grid.at(p),
                        position,
                        instances.at(p),
                        run,
                        spread,
                        divides,
                    ));
                    positions.push(position);
                    match comptime!(spread) {
                        Spread::Contiguous => scales.push(run),
                        Spread::Interleaved => scales.push(instances.at(p)),
                    }
                }
            }
        }

        (counts, positions, scales)
    }

    /// This walk taking one step along `dimension`, at the coordinate `coord`, counted in the
    /// level's tiles; a coordinate past the last takes the last.
    ///
    /// # Assumptions
    ///
    /// - `dimension` is a dimension of this walk's space.
    ///
    /// # Panics
    ///
    /// - Where `dimension` is not one of this walk's space, as the kernel expands.
    pub fn routed(self, #[comptime] dimension: Dimension, coord: usize) -> Walk {
        let rank = comptime!(self.space.rank());
        let at = comptime!(self.space.position(dimension));
        comptime!(assert!(
            self.space.contains(dimension),
            "Walk::routed: {dimension:?} is not a dimension of this walk's space, so it has no \
             coordinate to state"
        ));

        let mut counts = Coords::<usize>::new();
        let mut route = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            // Clamped: a data-derived coordinate may be out of range, and a panic here goes unseen.
            if p == at {
                // The dimension's own tiles, not `counts`, which is this instance's share.
                let last = comptime!(self.level.tiles(&self.space, dimension) - 1).runtime();
                counts.push(1usize);
                route.push(coord.min_with(last).cast::<u32>());
            } else {
                counts.push(self.counts.at(p));
                route.push(self.route.at(p));
            }
        }
        let steps = counts.product(comptime!((0..rank).collect::<Vec<_>>()));

        Walk {
            tiles: self.tiles,
            counts,
            positions: self.positions,
            scales: self.scales,
            route,
            routed_at: comptime!({
                let mut routed_at = self.routed_at.clone();
                routed_at.push(at);
                routed_at
            }),
            base: self.base,
            steps,
            ranged: comptime!(self.ranged),
            parent: self.parent,
            distributes: comptime!(self.distributes.clone()),
            space: comptime!(self.space.clone()),
            level: comptime!(self.level.clone()),
            unroll: comptime!(self.unroll),
            bounds: comptime!(None),
            run: comptime!(self.run),
        }
    }

    /// Who the stages filled along this walk belong to ([`StageOwner`]): each plane, where the
    /// walk sits under a level handing every plane a region of its own, or hands planes their own
    /// regions itself; the cube otherwise.
    pub(crate) fn stage_owner(&self) -> comptime_type!(StageOwner) {
        comptime!(StageOwner::new(
            self.parent
                .path
                .scope()
                .min(self.level.coverage().scope().unwrap_or(ComputeScope::Cube)),
            self.parent.path.planes_per_cube(),
            self.level.fillers(),
        ))
    }

    /// The region count.
    pub fn total(&self) -> usize {
        self.steps
    }

    /// Whether this instance takes turn `t` of a bounded walk: a constant for a turn every
    /// instance takes ([`takes_step`](Walk::takes_step) otherwise).
    pub(crate) fn takes_turn(&self, #[comptime] t: usize) -> bool {
        match comptime!(self.bounds) {
            Some((sure, _)) if t >= sure => self.takes_step(t),
            _ => true.runtime(),
        }
    }

    /// The line a copy walk ([`Walk::copy`]) hands this instance at turn `t`.
    pub(crate) fn copied_line(&self, t: usize) -> usize {
        self.region(t).copied_line()
    }

    /// Whether this instance takes step `i` of a bounded walk ([`bounds`](Walk::bounds)): whether
    /// the tile it reaches along the dimension its units take in turns lies inside the dimension's
    /// tiles.
    pub(crate) fn takes_step(&self, i: usize) -> bool {
        let p = comptime!(Walk::turns_at(&self.space, &self.level));
        self.tile_coord(self.step_coord(i, p), p) < self.tiles.at(p)
    }

    /// A walk whose units take its one dimension in turns, as a stride: this unit's first tile, the
    /// dimension's tiles, and how many units take them.
    #[allow(dead_code)] // Reached through its expand, from iterating a striding walk.
    pub(crate) fn stride(&self) -> (usize, usize, usize) {
        (self.positions.at(0), self.tiles.at(0), self.scales.at(0))
    }

    /// The region of a striding walk at its dimension's tile `tile`.
    #[allow(dead_code)] // Reached through its expand, from iterating a striding walk.
    pub(crate) fn region_at_tile(&self, tile: usize) -> Region {
        let mut coords = Coords::<u32>::new();
        coords.push(tile.cast::<u32>());
        self.parent.below(coords, &self.level)
    }

    /// This instance's position among those the level distributes dimension `p` to, `0` for a
    /// walked dimension.
    pub(crate) fn position_at(&self, #[comptime] p: usize) -> usize {
        self.positions.at(p)
    }

    /// This instance's place among the workers the level deals `dimension` to, `0` for a walked
    /// dimension: which plane of those taking `dimension` in runs, where the tiles a worker walks
    /// no longer name it.
    pub fn position(&self, #[comptime] dimension: Dimension) -> usize {
        self.positions.at(comptime!(self.space.position(dimension)))
    }

    /// The `i`-th region of the walk.
    pub fn region(&self, i: usize) -> Region {
        let idx = self.base.plus(i);
        self.parent.below(self.resolve(idx), &self.level)
    }

    /// The planes below this walk's regions, each once, at its first step ([`Region::planes`]):
    /// for a walk with no single region above it, such as a run of interleaved steps.
    pub fn planes(&self) -> Planes {
        self.region(0).planes()
    }

    /// Unravel a runtime step `idx` to its per-dimension coordinates.
    fn resolve(&self, idx: usize) -> Coords<u32> {
        let mut coords = Coords::<u32>::new();

        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            if comptime!(self.routed_at.contains(&p)) {
                coords.push(self.route.at(p));
            } else {
                coords.push(self.tile_coord(self.step_coord(idx, p), p).cast::<u32>());
            }
        }
        coords
    }

    /// The coordinate of step `idx` along dimension `p` (last dimension fastest).
    fn step_coord(&self, idx: usize, #[comptime] p: usize) -> usize {
        let rank = comptime!(self.space.rank());
        let quot = idx.divided_by(
            self.counts
                .product(comptime!(((p + 1)..rank).collect::<Vec<_>>())),
        );
        // `% count` is skipped when `idx` has no more significant coordinate, which constant
        // evaluation cannot know; except for a count of one, where `% 1` is `0`.
        let count = self.counts.at(p);
        let one = count.constant();
        let earlier = self
            .counts
            .product(comptime!((0..p).collect::<Vec<_>>()))
            .constant();
        if comptime!(one != Some(1) && earlier == Some(1)) {
            quot
        } else {
            quot.remainder(count)
        }
    }

    /// The tile dimension `p`'s step coordinate `coord` reaches, with the instance's position added
    /// per the [`Spread`], counted from the last tile where the level hands the dimension out
    /// [`reversed`](crate::Levels::reversed).
    fn tile_coord(&self, coord: usize, #[comptime] p: usize) -> usize {
        let tile = match comptime!(self.distributes[p].clone()) {
            DimensionDistribution::Walked => coord,
            DimensionDistribution::Distributed(Distribution {
                spread: Spread::Contiguous,
                ..
            }) => coord.plus(self.positions.at(p).times(self.scales.at(p))),
            DimensionDistribution::Distributed(Distribution {
                spread: Spread::Interleaved,
                ..
            }) => coord.times(self.scales.at(p)).plus(self.positions.at(p)),
        };
        let tiles = self.tiles.at(p);
        // A plane past the last tile stays past it, where the binding masks it.
        if comptime!(self.level.is_reversed(self.space.dimension_at(p))) && tile < tiles {
            tiles.minus(1usize).minus(tile)
        } else {
            tile
        }
    }
}

/// `for region in walk` visits the regions in order.
impl IntoIterator for Walk {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        let mut regions = Vec::new();
        for i in 0..self.total() {
            regions.push(self.region(i));
        }
        regions.into_iter()
    }
}

impl Iterable for WalkExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        // A bounded walk writes out its most steps, and guards those not every instance takes.
        if let (true, None, Some((sure, most))) = (self.unroll, self.const_len(), self.bounds) {
            for i in 0..most {
                let step = i.into_expand(scope);
                match i < sure {
                    true => body(scope, self.__expand_region_method(scope, step)),
                    false => {
                        let taken = self.__expand_takes_step_method(scope, step);
                        if_expand(scope, taken, |scope| {
                            body(scope, self.__expand_region_method(scope, step))
                        });
                    }
                }
            }
            return;
        }
        // Units taking the one dimension in turns, not written out: stride from the unit's place,
        // which is what taking every `n`-th tile is, with no count of its own to reckon.
        let whole = !self.ranged && self.routed_at.is_empty();
        if !self.unroll && whole && Walk::strides(&self.space, &self.level) {
            let (first, lines, units) = self.__expand_stride_method(scope);
            range_stepped::expand(scope, first, lines, units).expand(scope, &mut |scope, tile| {
                body(scope, self.__expand_region_at_tile_method(scope, tile));
            });
            return;
        }
        let range = RangeExpand::new(start, total);
        if self.unroll {
            range.expand_unroll(scope, &mut |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        } else {
            range.expand(scope, &mut |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        }
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        RangeExpand::new(start, total).expand_unroll(scope, &mut |scope, i| {
            body(scope, self.__expand_region_method(scope, i));
        });
    }

    /// The region count when it is a constant, so a single region needs no loop.
    fn const_len(&self) -> Option<usize> {
        crate::algebra::constant(&self.steps).map(|n| n as usize)
    }
}

impl Walk {
    /// The most turns a copy's walk writes out one after another ([`copy`](Walk::copy)): past
    /// it, a loop strides by the units. A runtime stride stalls each store on Metal's in-order
    /// pipe; a long run written out bloats the kernel.
    const MOST_WRITTEN_TURNS: usize = 8;

    /// Whether a copy of `lines` lines that `level` deals writes its turns out: where each unit
    /// takes a known few of them, a known count of lines dealt to a known count of units.
    pub(crate) fn copy_turns_written(level: &Level, lines: Option<usize>) -> bool {
        lines
            .and_then(|lines| level.copy_turns(lines))
            .is_some_and(|turns| turns <= Walk::MOST_WRITTEN_TURNS)
    }

    /// The position of the one dimension `level`'s units take in turns ([`Count::InTurns`]): such a
    /// level cuts that dimension alone.
    fn turns_at(space: &Space, level: &Level) -> usize {
        (0..space.rank())
            .find(|&p| {
                matches!(
                    level.count(space.dimension_at(p)),
                    Some(Count::InTurns { .. })
                )
            })
            .expect("Walk: a walk that guards its turns should be one whose units take a dimension in turns")
    }

    /// Whether this walk's units take the one dimension of its space in turns
    /// ([`Count::InTurns`]): walked as a stride from the unit's place, not counted through its
    /// steps.
    fn strides(space: &Space, level: &Level) -> bool {
        space.rank() == 1
            && matches!(
                level.count(space.dimension_at(0)),
                Some(Count::InTurns { .. })
            )
    }

    /// The steps every instance of `level` over `space` takes and the most any takes, where they
    /// differ only by how a dimension taken in turns splits among its units and the host knows
    /// both: the one walk whose instance's step count is not a constant but is bounded by one.
    fn bounds(space: &Space, level: &Level) -> Option<(usize, usize)> {
        let mut bounds = (1usize, 1usize);
        for dimension in space.dimensions() {
            let tiles = level.tiles_const(space, dimension)?;
            let (sure, most) = match level.count(dimension) {
                Some(Count::InTurns { units, .. }) => {
                    let units = units?;
                    (tiles / units, tiles.div_ceil(units))
                }
                _ if level.distributes(dimension) => return None,
                _ => (tiles, tiles),
            };
            bounds = (bounds.0 * sure, bounds.1 * most);
        }
        (bounds.0 != bounds.1).then_some(bounds)
    }
}

/// The settings a walk is told after it is built.
impl Walk {
    /// This walk, unrolled when iterated (static spaces only).
    pub fn unrolled(self) -> Walk {
        unexpanded!()
    }

    /// This walk, which steps `dimension` alone and in order, over the run its operands are cut to
    /// along it ([`Tile::within`](crate::Tile::within) from the walk's first region): every
    /// region but the last lies inside the run, so a ring over it
    /// ([`Stages::walk`](crate::Stages::walk)) fills those unchecked and bounds only the last, as
    /// over a run an index places ([`Region::walk_within`](crate::Region::walk_within)). A
    /// [`range`](Self::range) of it keeps the run.
    ///
    /// # Panics
    ///
    /// At expansion, where the walk's level steps another dimension than `dimension`, or steps it
    /// back.
    pub fn over_run(self, _dimension: Dimension) -> Walk {
        unexpanded!()
    }

    /// This walk over the `steps` regions starting at flat step `base`.
    /// The caller must keep `base + steps` within [`total`](Walk::total); nothing checks it.
    pub fn range(self, _base: usize, _steps: usize) -> Walk {
        unexpanded!()
    }

    /// The depth of the regions this walk hands out.
    pub(crate) fn depth(&self) -> usize {
        self.parent.path.depth() + 1
    }
}

impl WalkExpand {
    pub fn __expand_unrolled_method(mut self, _scope: &Scope) -> Self {
        self.unroll = true;
        self
    }

    pub fn __expand_over_run_method(mut self, _scope: &Scope, dimension: Dimension) -> Self {
        assert!(
            self.level.dimensions() == [dimension] && !self.level.is_reversed(dimension),
            "Walk::over_run: a walk over a run along {dimension:?} steps that dimension alone, in order"
        );
        self.run = Some(dimension);
        self
    }

    pub fn __expand_range_method(
        mut self,
        _scope: &Scope,
        base: NativeExpand<usize>,
        steps: NativeExpand<usize>,
    ) -> Self {
        self.base = base;
        self.steps = steps;
        self.ranged = true;
        self
    }

    pub(crate) fn depth(&self) -> usize {
        self.parent.path.depth() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(lines: usize, units: Option<usize>) -> Option<(usize, usize)> {
        Walk::bounds(
            &Space::new(&[(Level::COPIED, lines)]),
            &Level::copied_by(ComputeScope::Cube, units),
        )
    }

    /// Lines dealt to a known count of units that does not divide them: every unit takes the
    /// whole turns, some one more.
    #[test]
    fn a_ragged_deal_is_bounded_by_its_whole_and_its_last_turn() {
        assert_eq!(bounds(200, Some(64)), Some((3, 4)));
        assert_eq!(bounds(1, Some(64)), Some((0, 1)));
    }

    /// A copy whose every unit takes a known few turns writes them out.
    #[test]
    fn a_copy_of_a_few_known_turns_writes_them_out() {
        let level = Level::copied_by(ComputeScope::Cube, Some(64));
        assert!(Walk::copy_turns_written(&level, Some(256)));
        assert!(Walk::copy_turns_written(
            &level,
            Some(64 * Walk::MOST_WRITTEN_TURNS)
        ));
    }

    /// A copy of many turns, of an unknown count of lines, or dealt to an unknown count of units
    /// strides from the unit's place instead.
    #[test]
    fn a_copy_of_many_or_unknown_turns_strides() {
        let known = Level::copied_by(ComputeScope::Cube, Some(64));
        let unknown = Level::copied_by(ComputeScope::Cube, None);
        let many = 64 * (Walk::MOST_WRITTEN_TURNS + 1);
        assert!(!Walk::copy_turns_written(&known, Some(many)));
        assert!(!Walk::copy_turns_written(&known, None));
        assert!(!Walk::copy_turns_written(&unknown, Some(256)));
    }

    /// An even deal takes a constant count, and an unknown count of units bounds nothing.
    #[test]
    fn an_even_or_unknown_deal_has_no_bounds() {
        assert_eq!(bounds(256, Some(64)), None);
        assert_eq!(bounds(256, None), None);
    }
}
