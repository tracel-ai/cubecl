//! [`Levels`]: stating a partitioning's levels from the leaf up, each as a count of the one below.
//!
//! ```ignore
//! Levels::leaf(&[(M, 16), (N, 8), (K, 16)])   // the instruction
//!     .walk(&[(M, 2), (N, 4)])                // fragments a plane holds
//!     .walk(&[(K, 2)])                        // instruction steps in one stage
//!     .planes(&[(M, 2), (N, 4)])              // planes a cube holds
//!     .walk_every(&[K])                       // every stage `k` holds
//!     .cubes(&[M, N]).batches(&[B])           // every box `m x n` holds
//!     .build()
//! ```
//!
//! An `every` level ([`Count::All`]) closes its axis; only [`across`](Levels::across) may name it
//! again, on the cube level.

use super::{CubeOrder, PlaneOrder, UnitCube};
use crate::{Axis, ComputeScope, Count, Coverage, Level, Spread};

/// One level as the builder holds it before it is a [`Level`].
#[derive(Clone, Debug)]
struct StatedLevel {
    coverage: Coverage,
    /// `(axis, tile size, count)`; an every-level's count is unused.
    tiles: Vec<(Axis, usize, usize)>,
    takes: Takes,
    /// A cube level distributing one of its axes across several cubes.
    across: Option<(Axis, usize)>,
    shared_by: Option<usize>,
    fillers: usize,
    batches: Vec<Axis>,
    interleaved: Vec<Axis>,
    reversed: Vec<Axis>,
    /// Closed axes this cube level names again, to distribute across cubes.
    reopened: Vec<Axis>,
    order: CubeOrder,
    /// The order the launch's planes take this level's tiles in, where it spans the launch.
    plane_order: Option<PlaneOrder>,
    /// The cubes the launch's units are grouped in, where a unit level spans the launch.
    unit_cube: Option<UnitCube>,
}

/// Which [`Count`] a stated level's entries carry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Takes {
    Stated,
    Every,
    DistributedToUnits,
}

/// A partitioning stated from the leaf up.
#[derive(Clone, Debug)]
pub struct Levels {
    /// What one tile of each axis covers so far, leaf up.
    sizes: Vec<(Axis, usize)>,
    /// Axes an `every` level closed.
    closed: Vec<Axis>,
    /// Innermost first; [`build`](Self::build) reverses.
    stated: Vec<StatedLevel>,
}

impl Levels {
    /// The tile one worker holds at the bottom, e.g. an instruction's shape.
    pub fn leaf(tile: &[(Axis, usize)]) -> Self {
        Levels {
            sizes: tile.to_vec(),
            closed: Vec::new(),
            stated: Vec::new(),
        }
    }

    /// Every worker steps through this many of the level below.
    pub fn walk(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Coverage::Walk, counts)
    }

    /// Every worker steps through every tile of the level below along `axes`.
    pub fn walk_every(self, axes: &[Axis]) -> Self {
        self.every(Coverage::Walk, axes)
    }

    /// This many of the thing below, one per unit of the plane.
    pub fn units(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Coverage::Distribute(ComputeScope::Unit), counts)
    }

    /// This many of the level below, taken in turns by however many units the launch runs.
    pub fn units_distributed(mut self, axis: Axis, count: usize) -> Self {
        let tiles = vec![(axis, self.size(axis), count)];
        let mut stated = StatedLevel::new(
            Coverage::Distribute(ComputeScope::Unit),
            tiles,
            Takes::DistributedToUnits,
        );
        stated.interleaved.push(axis);
        self.stated.push(stated);
        self.grow(axis, count);
        self
    }

    /// This many of the level below, one per plane of the cube.
    pub fn planes(self, counts: &[(Axis, usize)]) -> Self {
        self.state(Coverage::Distribute(ComputeScope::Plane), counts)
    }

    /// This many of the level below, one per group of `planes` consecutive planes of the cube: an
    /// instruction the group's planes issue together, a warpgroup MMA's four, takes each tile.
    pub fn plane_groups(self, planes: usize, counts: &[(Axis, usize)]) -> Self {
        assert!(
            planes > 1,
            "Levels::plane_groups: a group of {planes} plane is a plane; state `planes`"
        );
        self.state(
            Coverage::Distribute(ComputeScope::PlaneGroup { planes }),
            counts,
        )
    }

    /// Every tile of the level below along `axis`, dealt to `planes` planes of the cube in runs:
    /// the planes need not divide the tiles, and a plane past the last whole run takes the rest.
    /// What splits a walk between the planes where its steps have no factor the planes fit.
    pub fn planes_across(self, axis: Axis, planes: usize) -> Self {
        let mut levels = self.every(Coverage::Distribute(ComputeScope::Plane), &[axis]);
        levels.last("planes_across").across = Some((axis, planes));
        levels
    }

    /// One cube for every box these axes hold; the entries ride the grid's `X`, `Y`, `Z` in order.
    pub fn cubes(self, axes: &[Axis]) -> Self {
        assert!(
            axes.len() <= 3,
            "Levels::cubes: {} axes, but a launch grid has three dimensions",
            axes.len()
        );
        self.every(Coverage::Distribute(ComputeScope::Cube), axes)
    }

    /// One plane of the launch for every tile these axes hold, the planes taking them in `order`:
    /// the outermost level of a partitioning with no cube level. A plane routine's planes share
    /// nothing, so which cube a plane sits in is the launch's grouping, not a level of the
    /// partitioning; the order sets the cubes' size ([`PlaneOrder::planes_per_cube`]).
    pub fn launch_planes(mut self, axes: &[Axis], order: PlaneOrder) -> Self {
        self = self.every(Coverage::Distribute(ComputeScope::Plane), axes);
        let stated = self.last("launch_planes");
        assert!(
            stated.reopened.is_empty(),
            "Levels::launch_planes: {:?} was taken whole by a level below",
            stated.reopened
        );
        stated.plane_order = Some(order);
        self
    }

    /// One unit of the launch for every tile these axes hold, in row order, the units grouped in
    /// cubes of `cube`: the outermost level of a partitioning with no cube or plane level. A unit
    /// routine's units share nothing, so which cube or plane a unit sits in is the launch's
    /// grouping, not a level of the partitioning, and `for unit in &partitioning` walks them.
    pub fn launch_units(mut self, axes: &[Axis], cube: UnitCube) -> Self {
        self = self.every(Coverage::Distribute(ComputeScope::Unit), axes);
        let stated = self.last("launch_units");
        assert!(
            stated.reopened.is_empty(),
            "Levels::launch_units: {:?} was taken whole by a level below",
            stated.reopened
        );
        stated.unit_cube = Some(cube);
        self
    }

    /// Distribute `axis` of the cube level just stated across `cubes` cubes, each a run (split-K).
    pub fn across(mut self, axis: Axis, cubes: usize) -> Self {
        let stated = self.last("across");
        assert!(
            stated.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Levels::across: only cubes take an axis in runs; the level just stated is {:?}",
            stated.coverage
        );
        assert!(
            stated.tiles.iter().any(|&(a, _, _)| a == axis),
            "Levels::across: {axis:?} is not an axis of the cube level just stated; name it there"
        );
        stated.across = Some((axis, cubes));
        self
    }

    /// Distribute this level's tiles to `cubes` cubes as one flat index (stream-K).
    pub fn shared_by(mut self, cubes: usize) -> Self {
        self.last("shared_by").shared_by = Some(cubes);
        self
    }

    /// The order the cube level just stated distributes its boxes to the grid.
    pub fn ordered(mut self, order: CubeOrder) -> Self {
        let stated = self.last("ordered");
        assert!(
            stated.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Levels::ordered: only cubes are distributed to a grid; the level just stated is {:?}",
            stated.coverage
        );
        stated.order = order.canonicalize();
        self
    }

    /// This walk's stages are filled by `n` extra planes that take no tile.
    pub fn filled_by(mut self, n: usize) -> Self {
        self.last("filled_by").fillers = n;
        self
    }

    /// Batch axes this cube level hands out one of.
    pub fn batches(mut self, axes: &[Axis]) -> Self {
        self.last("batches").batches.extend_from_slice(axes);
        self
    }

    /// The workers of the level just stated take `axis`'s tiles in turns rather than in runs.
    pub fn interleaved(mut self, axis: Axis) -> Self {
        self.last("interleaved").interleaved.push(axis);
        self
    }

    /// The level just stated, a cube level or a plane level spanning the launch, hands out
    /// `axis`'s tiles from the last to the first: the first workers launched take the last tiles.
    /// Where the later tiles hold more work, the heaviest start first and the light ones fill in
    /// behind them.
    pub fn reversed(mut self, axis: Axis) -> Self {
        self.last("reversed").reversed.push(axis);
        self
    }

    /// The levels, outermost first.
    pub fn build(self) -> Vec<Level> {
        for stated in &self.stated {
            for &axis in &stated.reopened {
                assert!(
                    matches!(stated.across, Some((a, _)) if a == axis),
                    "Levels::cubes: {axis:?} was taken whole by a level below; a cube level names \
                     it again only to distribute it across cubes (`.across({axis:?}, n)`)"
                );
            }
        }
        let groups = self
            .stated
            .iter()
            .filter(|stated| {
                matches!(
                    stated.coverage,
                    Coverage::Distribute(ComputeScope::PlaneGroup { .. })
                )
            })
            .count();
        let planes = self
            .stated
            .iter()
            .any(|stated| stated.coverage == Coverage::Distribute(ComputeScope::Plane));
        assert!(
            groups == 0 || (groups == 1 && !planes),
            "Levels: a cube's planes are dealt to plane groups by one level, beside no level of \
             single planes; these levels state {groups} plane group level(s){}",
            if planes { " and a plane level" } else { "" }
        );
        self.stated.iter().rev().map(StatedLevel::level).collect()
    }

    /// The single level this tiling states. Panics if it states several.
    pub fn level(self) -> Level {
        assert!(
            self.stated.len() == 1,
            "Levels::level: {} levels are stated; a walk over a region takes exactly one",
            self.stated.len()
        );
        self.build().remove(0)
    }

    /// State a level of `counts` of the level below, then grow each axis by its count.
    fn state(mut self, coverage: Coverage, counts: &[(Axis, usize)]) -> Self {
        let tiles = counts
            .iter()
            .map(|&(axis, count)| (axis, self.size(axis), count));
        self.stated
            .push(StatedLevel::new(coverage, tiles.collect(), Takes::Stated));
        for &(axis, count) in counts {
            self.grow(axis, count);
        }
        self
    }

    /// State a level covering every tile these axes hold, and close them.
    fn every(mut self, coverage: Coverage, axes: &[Axis]) -> Self {
        let reopened: Vec<Axis> = axes
            .iter()
            .copied()
            .filter(|axis| self.closed.contains(axis))
            .collect();
        if let (Coverage::Walk, Some(axis)) = (coverage, reopened.first()) {
            panic!(
                "Levels::walk_every: {axis:?} was taken whole by a level below, so a walk above \
                 it has nothing to step through"
            );
        }
        let tiles = axes.iter().map(|&axis| (axis, self.size(axis), 1));
        let mut stated = StatedLevel::new(coverage, tiles.collect(), Takes::Every);
        stated.reopened = reopened;
        self.stated.push(stated);
        self.closed.extend_from_slice(axes);
        self
    }

    /// What one tile of `axis` covers so far; `1` if unbuilt.
    fn size(&self, axis: Axis) -> usize {
        self.sizes
            .iter()
            .find(|&&(a, _)| a == axis)
            .map_or(1, |&(_, size)| size)
    }

    fn grow(&mut self, axis: Axis, count: usize) {
        assert!(
            !self.closed.contains(&axis),
            "Levels: {axis:?} was taken whole by a level below, so nothing above it may state a \
             count of its tiles"
        );
        match self.sizes.iter_mut().find(|(a, _)| *a == axis) {
            Some((_, size)) => *size *= count,
            None => self.sizes.push((axis, count)),
        }
    }

    fn last(&mut self, verb: &str) -> &mut StatedLevel {
        self.stated
            .last_mut()
            .unwrap_or_else(|| panic!("Levels::{verb}: no level has been stated for it to modify"))
    }
}

impl StatedLevel {
    fn new(coverage: Coverage, tiles: Vec<(Axis, usize, usize)>, takes: Takes) -> Self {
        StatedLevel {
            coverage,
            tiles,
            takes,
            across: None,
            shared_by: None,
            fillers: 0,
            batches: Vec::new(),
            interleaved: Vec::new(),
            reversed: Vec::new(),
            reopened: Vec::new(),
            order: CubeOrder::RowMajor,
            plane_order: None,
            unit_cube: None,
        }
    }

    /// How the level's workers take `axis`'s tiles.
    fn spread(&self, axis: Axis) -> Spread {
        match self.interleaved.contains(&axis) {
            true => Spread::Interleaved,
            false => Spread::Contiguous,
        }
    }

    /// The [`Level`] this states.
    fn level(&self) -> Level {
        let tile = || -> Vec<_> {
            self.tiles
                .iter()
                .map(|&(axis, tile, _)| (axis, tile))
                .collect()
        };
        if let Some(order) = &self.plane_order {
            return Level::spanning_launch(&tile(), order.clone()).reversed(&self.reversed);
        }
        if let Some(cube) = self.unit_cube {
            return Level::units_spanning_launch(&tile(), cube).reversed(&self.reversed);
        }
        let cuts: Vec<(Axis, usize, Count, Spread)> = self
            .tiles
            .iter()
            .map(|&(axis, tile, count)| {
                let count = match (self.takes, self.across) {
                    (Takes::Every, Some((across, cubes))) if across == axis => {
                        Count::AllAcross(cubes)
                    }
                    (Takes::Every, _) => Count::All,
                    (Takes::Stated, _) => Count::Stated(count),
                    (Takes::DistributedToUnits, _) => Count::Distributed(count),
                };
                (axis, tile, count, self.spread(axis))
            })
            .collect();
        let level = Level::new(self.coverage, &cuts);
        let level = match self.batches.is_empty() {
            true => level,
            false => level.batching(&self.batches),
        };
        let level = match self.fillers {
            0 => level,
            n => level.filling(n),
        };
        let level = match self.order.swizzles() {
            false => level,
            true => level.distributed_in(self.order),
        };
        let level = match self.shared_by {
            None => level,
            Some(cubes) => level.sharing(cubes),
        };
        match self.reversed.is_empty() {
            true => level,
            false => level.reversed(&self.reversed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Partitioning, Space};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);
    const B: Axis = Axis(3);

    /// A level's tiles stay inside the box it is handed exactly where its tile divides that box,
    /// which is what lets a window cut by it keep the page of the window it was cut from: pages of
    /// 16 keys over 20 overhang, and steps of 4 inside a page's box of 16 do not.
    #[test]
    fn a_level_overhangs_only_a_box_its_tile_does_not_divide() {
        let levels = Levels::leaf(&[(K, 4)])
            .walk(&[(K, 4)])
            .walk_every(&[K])
            .build();
        let keys = Space::new(&[(K, 20)]);
        assert!(levels[0].overhangs(&keys, K));
        let page = levels[0].child(&keys);
        assert_eq!(page.extent(K), 16);
        assert!(!levels[1].overhangs(&page, K));
    }

    /// Every level's tile is the product of what was stated below it.
    #[test]
    fn every_level_is_a_count_of_the_level_before() {
        let levels = Levels::leaf(&[(M, 16), (N, 8), (K, 16)])
            .walk(&[(M, 2), (N, 4)])
            .walk(&[(K, 2)])
            .planes(&[(M, 2), (N, 4)])
            .walk_every(&[K])
            .cubes(&[M, N])
            .batches(&[B])
            .build();
        assert_eq!(levels.len(), 5);
        let tiles = |i: usize, axis| levels[i].tile(axis);
        let counts = |i: usize, axis| levels[i].count(axis);

        assert_eq!((tiles(0, M), tiles(0, N)), (Some(64), Some(128)));
        assert_eq!(
            (counts(0, M), counts(0, N)),
            (Some(Count::All), Some(Count::All))
        );
        assert_eq!((tiles(0, B), counts(0, B)), (Some(1), Some(Count::All)));
        assert_eq!(tiles(0, K), None);

        assert_eq!((tiles(1, K), counts(1, K)), (Some(32), Some(Count::All)));
        assert_eq!(
            (tiles(2, M), counts(2, M)),
            (Some(32), Some(Count::Stated(2)))
        );
        assert_eq!(
            (tiles(2, N), counts(2, N)),
            (Some(32), Some(Count::Stated(4)))
        );
        assert_eq!(
            (tiles(3, K), counts(3, K)),
            (Some(16), Some(Count::Stated(2)))
        );
        assert_eq!(
            (tiles(4, M), counts(4, M)),
            (Some(16), Some(Count::Stated(2)))
        );
        assert_eq!(
            (tiles(4, N), counts(4, N)),
            (Some(8), Some(Count::Stated(4)))
        );
    }

    /// The planes take a walk's steps in runs at a count that need not divide them: the plane
    /// level states the count of planes, not a tile of planes times a step.
    #[test]
    fn the_planes_take_every_step_across_in_runs() {
        let levels = Levels::leaf(&[(N, 8), (K, 16)])
            .planes_across(K, 4)
            .cubes(&[N])
            .build();
        assert_eq!(
            levels[1].coverage(),
            Coverage::Distribute(ComputeScope::Plane)
        );
        assert_eq!(levels[1].count(K), Some(Count::AllAcross(4)));
        assert_eq!(levels[1].tile(K), Some(16));
    }

    /// Plane groups are dealt by one level, beside no level of single planes, which the cube's
    /// count of planes would otherwise take twice.
    #[test]
    #[should_panic(expected = "a cube's planes are dealt to plane groups by one level")]
    fn plane_groups_take_no_plane_level_beside_them() {
        Levels::leaf(&[(M, 64), (N, 128), (K, 64)])
            .planes(&[(N, 2)])
            .plane_groups(4, &[(M, 2)])
            .walk_every(&[K])
            .build();
    }

    /// A plane group's level counts its groups, and the cube holds each group's planes.
    #[test]
    fn a_plane_group_holds_its_planes() {
        let levels = Levels::leaf(&[(M, 64), (N, 128), (K, 64)])
            .plane_groups(4, &[(M, 2)])
            .walk_every(&[K])
            .filled_by(1)
            .cubes(&[M, N])
            .build();
        let groups = Coverage::Distribute(ComputeScope::PlaneGroup { planes: 4 });
        assert!(levels.iter().any(|level| level.coverage() == groups));
        let partitioning = Partitioning::new(Space::new(&[(M, 1024), (N, 1024), (K, 512)]), levels);
        assert_eq!(partitioning.instances(groups), 2);
        assert_eq!(partitioning.planes_per_cube(), 9);
    }

    /// A closed axis returns to the cube level to be split across cubes.
    #[test]
    fn a_closed_axis_returns_to_the_cube_level_to_be_distributed_across() {
        let levels = Levels::leaf(&[(M, 16), (N, 8), (K, 16)])
            .walk_every(&[K])
            .cubes(&[M, N, K])
            .across(K, 4)
            .build();
        assert_eq!(levels[0].count(K), Some(Count::AllAcross(4)));
        assert_eq!(levels[0].tile(K), Some(16));
        assert_eq!(levels[0].cube_axis(K), Some(crate::CubeAxis::Z));
        assert_eq!(levels[1].count(K), Some(Count::All));
    }

    /// A count builds its tile, so nothing can fail to divide.
    #[test]
    fn a_count_builds_its_tile_and_nothing_refuses() {
        let levels = Levels::leaf(&[(M, 5)])
            .planes(&[(M, 3)])
            .walk(&[(M, 7)])
            .cubes(&[M])
            .build();
        assert_eq!(levels[0].tile(M), Some(105));
        assert_eq!(levels[1].tile(M), Some(15));
        assert_eq!(levels[1].count(M), Some(Count::Stated(7)));
        assert_eq!(levels[2].count(M), Some(Count::Stated(3)));
        // Only the top every-level can overhang.
        let space = Space::new(&[(M, 100)]);
        assert!(levels[0].overhangs(&space, M));
        let cube = levels[0].child(&space);
        assert!(!levels[1].overhangs(&cube, M));
        assert!(!levels[2].overhangs(&levels[1].child(&cube), M));
    }

    /// Walking every tile is not walking one.
    #[test]
    fn every_is_not_one() {
        let every = Levels::leaf(&[(K, 16)]).walk_every(&[K]).level();
        let one = Levels::leaf(&[(K, 16)]).walk(&[(K, 1)]).level();
        assert_ne!(every, one);
        assert_eq!(every.count(K), Some(Count::All));
        assert_eq!(one.count(K), Some(Count::Stated(1)));
    }

    /// A count distributed to the units builds its tile like any other.
    #[test]
    fn a_count_distributed_to_the_units_builds_its_tile() {
        let levels = Levels::leaf(&[(M, 4), (N, 4)])
            .units_distributed(N, 64)
            .planes(&[(M, 2)])
            .cubes(&[M, N])
            .build();
        assert_eq!(levels[0].tile(N), Some(256));
        assert_eq!(levels[2].count(N), Some(Count::Distributed(64)));
        assert_eq!(
            levels[2].coverage(),
            Coverage::Distribute(ComputeScope::Unit)
        );
        assert_eq!(levels[2].spread(N), Some(Spread::Interleaved));
        let partitioning = Partitioning::new(Space::new(&[(M, 100), (N, 1000)]), levels);
        assert_eq!(partitioning.units(), 1);
        assert_eq!(partitioning.planes_per_cube(), 2);
    }

    #[test]
    #[should_panic(expected = "distributes no other axis to the units")]
    fn a_count_distributed_to_the_units_is_their_only_axis() {
        let _ = Level::new(
            Coverage::Distribute(ComputeScope::Unit),
            &[
                (M, 4, Count::Distributed(8), Spread::Interleaved),
                (N, 4, Count::Stated(4), Spread::Contiguous),
            ],
        );
    }

    /// A region walk is the chain it stands for.
    #[test]
    fn a_region_walk_is_stated_once() {
        assert_eq!(
            Level::every(&[(M, 8), (K, 16)]),
            Levels::leaf(&[(M, 8), (K, 16)]).walk_every(&[M, K]).level()
        );
    }

    #[test]
    #[should_panic(
        expected = "taken whole by a level below, so nothing above it may state a count"
    )]
    fn nothing_above_all_of_it_may_count_it() {
        let _ = Levels::leaf(&[(K, 16)]).walk_every(&[K]).walk(&[(K, 2)]);
    }

    #[test]
    #[should_panic(expected = "a walk above it has nothing to step through")]
    fn all_of_it_is_walked_once() {
        let _ = Levels::leaf(&[(K, 16)]).walk_every(&[K]).walk_every(&[K]);
    }

    #[test]
    #[should_panic(expected = "names it again only to distribute it across cubes")]
    fn a_closed_axis_on_the_cube_level_must_be_distributed_across() {
        let _ = Levels::leaf(&[(M, 16), (K, 16)])
            .walk_every(&[K])
            .cubes(&[M, K])
            .build();
    }

    #[test]
    #[should_panic(expected = "only cubes take an axis in runs")]
    fn only_cubes_take_an_axis_in_runs() {
        let _ = Levels::leaf(&[(K, 16)]).planes(&[(K, 4)]).across(K, 2);
    }

    /// A reversed axis reaches the cube level that states it, and no level below.
    #[test]
    fn a_reversed_axis_reaches_the_cube_level() {
        let levels = Levels::leaf(&[(M, 16), (N, 8), (K, 16)])
            .walk_every(&[K])
            .planes(&[(M, 2)])
            .cubes(&[M, N])
            .reversed(M)
            .build();
        assert!(levels[0].is_reversed(M));
        assert!(!levels[0].is_reversed(N));
        assert!(levels[1..].iter().all(|level| !level.is_reversed(M)));
    }

    /// The planes of one cube are not dealt in launch order, so there is nothing to reverse.
    #[test]
    #[should_panic(expected = "only a cube level or a level spanning the launch")]
    fn only_a_launch_order_is_reversed() {
        let _ = Levels::leaf(&[(M, 16)])
            .planes(&[(M, 2)])
            .reversed(M)
            .build();
    }

    #[test]
    #[should_panic(expected = "a walk over a region takes exactly one")]
    fn a_walk_over_a_region_takes_one_level() {
        let _ = Levels::leaf(&[(K, 16)])
            .walk(&[(K, 2)])
            .walk_every(&[K])
            .level();
    }
}
