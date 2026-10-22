//! The [`Walk`]: the regions one [`Level`] of a [`Space`] hands the instance running the code.

use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::{
    Axis, ComputeScope, Coords, Integer, IntegerExpand, Level, Planes, Region, RegionExpand, Space,
    StageOwner,
};

use super::distribution::{AxisDistribution, Distribution};
use crate::Spread;
use crate::space::partition::{GridCount, in_plane_axes, launch_tiles, swizzled_positions};

/// The runtime odometer over a [`Space`]'s tiles under one [`Level`].
#[derive(CubeType)]
pub struct Walk {
    /// The level's tiles along each axis, which a reversed axis counts back from.
    tiles: Coords<usize>,
    counts: Coords<usize>,
    positions: Coords<usize>,
    scales: Coords<usize>,
    /// Per-axis routed coordinate ([`routed`](Walk::routed)), `0` elsewhere.
    route: Coords<u32>,
    /// Which axes [`route`](Self::route) speaks for.
    #[cube(comptime)]
    routed_at: Vec<usize>,
    base: usize,
    steps: usize,
    pub(crate) parent: Region,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) level: Level,
    /// How the level distributes each axis of the space, in axis order.
    #[cube(comptime)]
    distributes: Vec<AxisDistribution>,
    /// Whether iterating this walk unrolls.
    #[cube(comptime)]
    pub(crate) unroll: bool,
    /// The axis this walk steps a run along ([`Region::walk_within`]), where it steps nothing
    /// else and in order: every region but the last then lies inside the run.
    #[cube(comptime)]
    pub(crate) run: Option<Axis>,
}

#[cube]
impl Walk {
    // The loops are `#[unroll]`ed cube loops over a comptime `Vec`, not host iteration.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn of(space: &Space, #[comptime] level: Level, parent: Region) -> Walk {
        let host = comptime!(space.clone());
        let rank = comptime!(host.rank());
        // A swizzled cube level decodes its two in-plane axes jointly from the flat dispatch index.
        let in_plane = comptime!(
            level
                .order()
                .swizzles()
                .then(|| in_plane_axes(&host, &level))
        );
        let distributes = comptime!(
            (0..rank)
                .map(|p| AxisDistribution::new(&level, &host, p, in_plane))
                .collect::<Vec<_>>()
        );

        let mut grid = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(level.grid(space.axis_at(p))) {
                GridCount::Const(n) => grid.push(n.runtime()),
                GridCount::Extent(tile) => grid.push(space.count(p, tile)),
            }
        }

        let (counts, positions, scales) = match comptime!(level.spans_launch()) {
            true => Walk::spanning_launch(comptime!(host.clone()), comptime!(level.clone()), &grid),
            false => Walk::dealt(
                comptime!(host.clone()),
                comptime!(level.clone()),
                comptime!(distributes.clone()),
                &grid,
            ),
        };

        // Folded, not accumulated, so a static walk can unroll.
        let steps = counts.product(comptime!((0..rank).collect::<Vec<_>>()));

        Walk {
            tiles: grid,
            counts,
            positions,
            scales,
            route: Coords::constant(comptime!(vec![0; rank])),
            routed_at: comptime!(Vec::new()),
            base: 0usize,
            steps,
            parent,
            distributes,
            space: host,
            level,
            unroll: comptime!(false),
            run: comptime!(None),
        }
    }

    /// `(counts, positions, scales)` of a level spanning the launch: along each of its axes, the
    /// one tile this plane's or unit's place in the launch names, past the last for one past the
    /// last tile, which the binding masks; every tile of the others.
    fn spanning_launch(
        #[comptime] host: Space,
        #[comptime] level: Level,
        grid: &Coords<usize>,
    ) -> (Coords<usize>, Coords<usize>, Coords<usize>) {
        let (positions, counts) = launch_tiles(comptime!(host.clone()), level, grid);
        let mut scales = Coords::<usize>::new();
        #[unroll]
        for _ in 0..comptime!(host.rank()) {
            scales.push(1usize);
        }
        (counts, positions, scales)
    }

    /// `(counts, positions, scales)` of a level dealt to the instances of one cube, or of the
    /// grid's cubes, each axis as `distributes` says.
    #[allow(clippy::needless_range_loop)]
    fn dealt(
        #[comptime] host: Space,
        #[comptime] level: Level,
        #[comptime] distributes: Vec<AxisDistribution>,
        grid: &Coords<usize>,
    ) -> (Coords<usize>, Coords<usize>, Coords<usize>) {
        let rank = comptime!(host.rank());
        // Folded, so a constant grid's decode folds too.
        let mut instances = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(distributes[p].clone()) {
                AxisDistribution::Distributed(Distribution {
                    across: Some(workers),
                    ..
                }) => instances.push(workers.runtime()),
                // The plane's width is the launch's, so it is read, not compiled in.
                AxisDistribution::Distributed(Distribution { in_turns: true, .. }) => {
                    instances.push(CUBE_DIM_X as usize)
                }
                AxisDistribution::Distributed(_) => instances.push(grid.at(p)),
                AxisDistribution::Walked => instances.push(1usize),
            }
        }
        let swizzled = swizzled_positions(
            comptime!(host.clone()),
            comptime!(level.clone()),
            &instances,
        );

        let mut counts = Coords::<usize>::new();
        let mut positions = Coords::<usize>::new();
        let mut scales = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(distributes[p].clone()) {
                AxisDistribution::Walked => {
                    counts.push(grid.at(p));
                    positions.push(0usize);
                    scales.push(1usize);
                }
                AxisDistribution::Distributed(Distribution {
                    scope: compute_scope,
                    dim,
                    spread,
                    across,
                    in_turns: _,
                    divides,
                    inner,
                    unspanned,
                    swizzled: joint,
                }) => {
                    // Mixed-radix stride over the later same-dimension axes, spanned or not.
                    let inner_weight = instances.product(inner) * comptime!(unspanned).runtime();
                    let position = match comptime!(joint) {
                        Some(i) => swizzled.at(i),
                        None => AxisDistribution::hardware(compute_scope, dim)
                            .divided_by(inner_weight)
                            .remainder(instances.at(p)),
                    };
                    let run = match comptime!(across) {
                        Some(workers) => grid
                            .at(p)
                            .plus(comptime!(workers - 1).runtime())
                            .divided_by(workers.runtime()),
                        None => 1usize.runtime(),
                    };
                    counts.push(AxisDistribution::tiles(
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

    /// This walk taking one step along `axis`, at the coordinate `coord`.
    /// The caller must keep `coord` within the axis's extent; nothing checks it.
    pub fn routed(self, #[comptime] axis: Axis, coord: usize) -> Walk {
        let rank = comptime!(self.space.rank());
        let at = comptime!(self.space.position(axis));
        comptime!(assert!(
            self.space.contains(axis),
            "Walk::routed: {axis:?} is not an axis of this walk's space, so it has no \
             coordinate to state"
        ));

        let mut counts = Coords::<usize>::new();
        let mut route = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            // Clamped: a data-derived coordinate may be out of range, and a panic here goes unseen.
            if comptime!(p == at) {
                // The axis's own tiles, not `counts`, which is this instance's share.
                let last = comptime!(self.level.tiles(&self.space, axis) - 1).runtime();
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
            parent: self.parent,
            distributes: comptime!(self.distributes.clone()),
            space: comptime!(self.space.clone()),
            level: comptime!(self.level.clone()),
            unroll: comptime!(self.unroll),
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

    /// This instance's position among those the level distributes axis `p` to, `0` for a walked
    /// axis.
    pub(crate) fn position_at(&self, #[comptime] p: usize) -> usize {
        self.positions.at(p)
    }

    /// This instance's place among the workers the level deals `axis` to, `0` for a walked axis:
    /// which plane of those taking `axis` in runs, where the tiles a worker walks no longer name
    /// it.
    pub fn position(&self, #[comptime] axis: Axis) -> usize {
        self.positions.at(comptime!(self.space.position(axis)))
    }

    /// The `i`-th region of the walk.
    pub fn region(&self, i: usize) -> Region {
        let idx = self.base.plus(i);
        self.parent
            .below(self.resolve(idx), comptime!(self.level.clone()))
    }

    /// The planes below this walk's regions, each once, at its first step ([`Region::planes`]):
    /// for a walk with no single region above it, such as a run of interleaved steps.
    pub fn planes(&self) -> Planes {
        self.region(0).planes()
    }

    /// Unravel a runtime step `idx` to its per-axis coordinates.
    fn resolve(&self, idx: usize) -> Coords<u32> {
        let mut coords = Coords::<u32>::new();

        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            if comptime!(self.routed_at.contains(&p)) {
                coords.push(self.route.at(p));
            } else {
                coords.push(self.fold(self.digit(idx, p), p).cast::<u32>());
            }
        }
        coords
    }

    /// The odometer digit of step `idx` along axis `p` (last axis fastest).
    fn digit(&self, idx: usize, #[comptime] p: usize) -> usize {
        let rank = comptime!(self.space.rank());
        let quot = idx.divided_by(
            self.counts
                .product(comptime!(((p + 1)..rank).collect::<Vec<_>>())),
        );
        // `% count` is skipped when `idx` has no more significant digit, which folding cannot know;
        // except for a count of one, where `% 1` folds to `0`.
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

    /// Fold axis `p`'s instance position into its `digit` per the [`Spread`], counted from the
    /// last tile where the level hands the axis out [`reversed`](crate::Levels::reversed).
    fn fold(&self, digit: usize, #[comptime] p: usize) -> usize {
        let tile = match comptime!(self.distributes[p].clone()) {
            AxisDistribution::Walked => digit,
            AxisDistribution::Distributed(Distribution {
                spread: Spread::Contiguous,
                ..
            }) => digit.plus(self.positions.at(p).times(self.scales.at(p))),
            AxisDistribution::Distributed(Distribution {
                spread: Spread::Interleaved,
                ..
            }) => digit.times(self.scales.at(p)).plus(self.positions.at(p)),
        };
        let tiles = self.tiles.at(p);
        // A plane past the last tile stays past it, where the binding masks it.
        if comptime!(self.level.is_reversed(self.space.axis_at(p))) && tile < tiles {
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

    /// The region count when it folds to a constant, so a single region needs no loop.
    fn const_len(&self) -> Option<usize> {
        crate::algebra::constant(&self.steps).map(|n| n as usize)
    }
}

/// The settings a walk is told after it is built.
impl Walk {
    /// This walk, unrolled when iterated (static spaces only).
    pub fn unrolled(self) -> Walk {
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

    pub fn __expand_range_method(
        mut self,
        _scope: &Scope,
        base: NativeExpand<usize>,
        steps: NativeExpand<usize>,
    ) -> Self {
        self.base = base;
        self.steps = steps;
        self
    }

    pub(crate) fn depth(&self) -> usize {
        self.parent.path.depth() + 1
    }
}
