//! A [`Space`] and the [`Level`]s that cut it, held as one value.

use cubecl::{prelude::*, unexpanded};

use crate::{
    Axis, ComputeScope, Count, Coverage, CubeAxis, Level, LevelTable, Region, RegionExpand, Space,
    Walk,
};

use super::UnitCube;

/// A space with the levels that partition it, outermost first: what a kernel's loops iterate.
#[derive(CubeType, CubeLaunch, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partitioning {
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) levels: Vec<Level>,
}

/// The comptime partitioning of a runtime one.
impl PartitioningExpand {
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Partitioning {
        Partitioning::new(self.space.clone(), self.levels.clone())
    }

    pub fn space(&self) -> Space {
        self.space.clone()
    }
}

/// The comptime reads of a partitioning a kernel is handed.
impl PartitioningExpand {
    /// The levels, outermost first.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }
}

impl Partitioning {
    /// The levels, outermost first.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// Level `i`, outermost first.
    pub fn level(&self, i: usize) -> Level {
        self.levels[i].clone()
    }
}

#[cube]
impl Partitioning {
    /// The regions of the first level: what `for cube in space` iterates.
    pub fn walk(&self) -> Walk {
        Region::root(self).walk()
    }

    /// The regions of `level` over this space, a level of the kernel's own.
    pub fn over(&self, #[comptime] level: &Level) -> Walk {
        Walk::of(&self.space, comptime!(level.clone()), Region::root(self))
    }
}

/// Host-side stand-in for `for plane in space`; never runs.
impl IntoIterator for Partitioning {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

impl IntoIterator for &Partitioning {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// `for region in partitioning` iterates its first level.
impl Iterable for PartitioningExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, &self).expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, &self).expand_unroll(scope, body)
    }
}

/// `for cube in &space`: the same, leaving `space` to the body.
impl Iterable for &PartitioningExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, self).expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, self).expand_unroll(scope, body)
    }
}

impl Partitioning {
    /// `space` cut by `levels`, outermost first. Panics where two levels distribute over one
    /// scope: a scope's instances take one level's tiles.
    pub fn new(space: Space, levels: Vec<Level>) -> Self {
        for (i, level) in levels.iter().enumerate() {
            if let Some(scope) = level.coverage().scope() {
                assert!(
                    levels[i + 1..]
                        .iter()
                        .all(|other| other.coverage() != level.coverage()),
                    "Partitioning::new: two levels distribute over {scope:?}"
                );
            }
            if level.spans_launch() {
                assert!(
                    i == 0,
                    "Partitioning::new: a level spanning the launch is the outermost level, and \
                     this one sits {i} levels down"
                );
                assert!(
                    levels.iter().all(|other| other.coverage()
                        != Coverage::Distribute(ComputeScope::Cube)
                        && other.fillers() == 0),
                    "Partitioning::new: the launch's planes or units take the outermost level, so \
                     no cube level groups them and no plane only fills a cube's stages"
                );
            }
            if level.unit_cube().is_some() {
                assert!(
                    levels.iter().all(|other| !matches!(
                        other.coverage(),
                        Coverage::Distribute(ComputeScope::Plane | ComputeScope::PlaneGroup { .. })
                    )),
                    "Partitioning::new: the launch's units take the outermost level, so no plane \
                     level holds them"
                );
            }
        }
        Self { space, levels }
    }

    /// The space these levels cut.
    pub fn space(&self) -> &Space {
        &self.space
    }

    /// This partitioning with every extent `Dynamic`.
    pub fn all_dynamic(self) -> Self {
        Partitioning {
            space: self.space.all_dynamic(),
            levels: self.levels,
        }
    }

    /// This partitioning's table with a name for each axis; unnamed axes print their index.
    pub fn table<'a>(&'a self, labels: &'a [(Axis, &'a str)]) -> LevelTable<'a> {
        LevelTable::new(self, labels)
    }

    /// The leaf the levels reach.
    pub fn leaf(&self) -> Space {
        self.space.leaf(&self.levels)
    }

    /// Whether some level's edge fails to divide the extent of `axis` handed to it.
    pub fn overhangs(&self, axis: Axis) -> bool {
        assert!(
            !self.space.is_dynamic(axis),
            "Partitioning::overhangs: axis {axis:?} is Dynamic; ask the concrete space"
        );
        let mut space = self.space.clone();
        for level in &self.levels {
            if level.overhangs(&space, axis) {
                return true;
            }
            space = level.child(&space);
        }
        false
    }

    /// Every axis some tile reaches past the end of, whose accesses are masked, in the space's
    /// order.
    ///
    /// One walk down the levels, every axis asked at each: [`overhangs`](Self::overhangs) asked
    /// axis by axis would cut the space down through every level once per axis, and a launch asks
    /// this of every operand it binds, on every call.
    pub(crate) fn overhanging(&self) -> Vec<Axis> {
        let axes: Vec<Axis> = self.space.axes().collect();
        for &axis in &axes {
            assert!(
                !self.space.is_dynamic(axis),
                "Partitioning::overhanging: axis {axis:?} is Dynamic; ask the concrete space"
            );
        }
        let mut overhanging = vec![false; axes.len()];
        let mut space = self.space.clone();
        for level in &self.levels {
            for (hangs, &axis) in overhanging.iter_mut().zip(&axes) {
                *hangs |= level.overhangs(&space, axis);
            }
            space = level.child(&space);
        }
        axes.into_iter()
            .zip(overhanging)
            .filter_map(|(axis, hangs)| hangs.then_some(axis))
            .collect()
    }

    /// The one level distributed over `scope`'s workers, if any.
    pub fn level_distributed(&self, scope: ComputeScope) -> Option<&Level> {
        let coverage = Coverage::Distribute(scope);
        self.levels
            .iter()
            .find(|level| level.coverage() == coverage)
    }

    /// The box one of `scope`'s instances covers: the space cut down through the level distributed
    /// over it. Panics where no level distributes over `scope`.
    pub fn box_of(&self, scope: ComputeScope) -> Space {
        let coverage = Coverage::Distribute(scope);
        let mut space = self.space.clone();
        for level in &self.levels {
            space = level.child(&space);
            if level.coverage() == coverage {
                return space;
            }
        }
        panic!("Partitioning::box_of: no level distributes over {scope:?}")
    }

    /// The levels one instance walks, outermost first.
    pub fn walks(&self) -> impl Iterator<Item = &Level> + '_ {
        self.levels
            .iter()
            .filter(|level| level.coverage() == Coverage::Walk)
    }

    /// Instances `coverage` distributes the space to.
    pub(crate) fn instances(&self, coverage: Coverage) -> u32 {
        match coverage {
            Coverage::Distribute(ComputeScope::Cube) => [CubeAxis::X, CubeAxis::Y, CubeAxis::Z]
                .into_iter()
                .map(|dim| self.cube_instances(dim))
                .product(),
            Coverage::Distribute(ComputeScope::Plane)
            | Coverage::Distribute(ComputeScope::PlaneGroup { .. })
            | Coverage::Distribute(ComputeScope::Unit) => {
                self.count_instances(|level| level.coverage() == coverage, |_, _| true)
            }
            Coverage::Walk => 1,
        }
    }

    /// Cubes on grid dimension `dim`.
    fn cube_instances(&self, dim: CubeAxis) -> u32 {
        self.count_instances(
            |level| {
                level.coverage() == Coverage::Distribute(ComputeScope::Cube)
                    && (level.shared_by().is_none() || dim == CubeAxis::X)
            },
            |level, axis| level.cube_axis(axis) == Some(dim),
        )
    }

    /// The product over the selected levels of their instance counts; partial tiles count.
    fn count_instances(
        &self,
        levels: impl Fn(&Level) -> bool,
        axes: impl Fn(&Level, Axis) -> bool,
    ) -> u32 {
        let mut total = 1u32;
        let mut space = self.space.clone();
        for level in &self.levels {
            // A unit of a level spanning the launch takes one tile, and asks for no units of its own.
            if levels(level) && level.unit_cube().is_none() {
                match level.shared_by() {
                    Some(workers) => total *= workers as u32,
                    None => {
                        for axis in space.axes() {
                            if level.distributes(axis) && axes(level, axis) {
                                total *= match level.count(axis) {
                                    Some(Count::AllAcross(workers)) => workers,
                                    // Launch-sized units ask for none of their own.
                                    Some(Count::Distributed(_)) => 1,
                                    _ => level.tiles(&space, axis),
                                } as u32;
                            }
                        }
                    }
                }
            }
            space = level.child(&space);
        }
        total
    }

    /// The cube grid these levels distribute to.
    pub fn cube_count(&self) -> CubeCount {
        if let Some(order) = self.levels.first().and_then(Level::launch_order) {
            let (x, y) = order.grid(&self.levels[0], &self.space);
            return CubeCount::Static(x as u32, y as u32, 1);
        }
        CubeCount::Static(
            self.cube_instances(CubeAxis::X),
            self.cube_instances(CubeAxis::Y),
            self.cube_instances(CubeAxis::Z),
        )
    }

    /// Planes one cube holds: the levels' plane cuts, each plane group's planes, plus any that
    /// only fill; where the planes or units span the launch, the cubes their order groups them in.
    pub fn planes_per_cube(&self) -> u32 {
        if let Some(order) = self.levels.first().and_then(Level::plane_order) {
            return order.planes_per_cube() as u32;
        }
        if let Some(cube) = self.unit_cube() {
            return cube.planes as u32;
        }
        // `Levels` deals plane groups by one level, beside no plane level.
        let grouped = self.levels.iter().find_map(|level| {
            let scope = level.coverage().scope()?;
            let planes = scope.planes().filter(|_| scope != ComputeScope::Plane)?;
            Some(self.instances(level.coverage()) * planes as u32)
        });
        grouped.unwrap_or_else(|| self.instances(Coverage::Distribute(ComputeScope::Plane)))
            + self.fillers()
    }

    /// Planes the cube holds that fill a walk's stages and take no tile
    /// ([`Levels::filled_by`](crate::Levels::filled_by)).
    pub fn fillers(&self) -> u32 {
        self.levels.iter().map(|level| level.fillers() as u32).sum()
    }

    /// Units one instance holds; `1` where no level cuts to units or they take tiles in turns.
    pub fn units(&self) -> u32 {
        self.instances(Coverage::Distribute(ComputeScope::Unit))
    }

    /// The cube this partitioning asks for at `plane_size`; where the units span the launch, the
    /// cube they are grouped in, whatever `plane_size`.
    pub fn cube_dim(&self, plane_size: u32) -> CubeDim {
        match self.unit_cube() {
            Some(cube) => CubeDim::new_2d(cube.plane_width as u32, cube.planes as u32),
            None => CubeDim::new_2d(plane_size, self.planes_per_cube()),
        }
    }

    /// The cubes the launch's units are grouped in, where a unit level spans the launch.
    pub fn unit_cube(&self) -> Option<UnitCube> {
        self.levels.first().and_then(Level::unit_cube)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Levels;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A cube grid over `M`/`N`, a `K` walk, then one partition per plane.
    fn staged(fillers: usize) -> Partitioning {
        Partitioning::new(
            Space::new(&[(M, 256), (N, 256), (K, 512)]),
            Levels::leaf(&[(M, 64), (N, 64), (K, 64)])
                .planes(&[(M, 2), (N, 2)])
                .walk_every(&[K])
                .filled_by(fillers)
                .cubes(&[M, N])
                .build(),
        )
    }

    #[test]
    fn a_walk_with_no_fillers_is_the_partitioning_it_always_was() {
        let plain = staged(0);
        assert_eq!(plain.fillers(), 0);
        assert_eq!(plain.planes_per_cube(), 4);
        assert_eq!(plain.cube_dim(32), CubeDim::new_2d(32, 4));
    }

    /// Filling planes widen the cube and change nothing else.
    #[test]
    fn a_filling_plane_widens_the_cube_and_nothing_else() {
        let plain = staged(0);
        let filled = staged(1);

        assert_eq!(filled.planes_per_cube(), plain.planes_per_cube() + 1);
        assert_eq!(filled.cube_dim(32), CubeDim::new_2d(32, 5));

        assert_eq!(
            filled.instances(Coverage::Distribute(ComputeScope::Plane)),
            plain.instances(Coverage::Distribute(ComputeScope::Plane))
        );
        assert_eq!(filled.units(), plain.units());
        for dim in [CubeAxis::X, CubeAxis::Y, CubeAxis::Z] {
            assert_eq!(filled.cube_instances(dim), plain.cube_instances(dim));
        }
    }

    #[test]
    fn the_fillers_of_every_level_are_counted() {
        assert_eq!(staged(3).fillers(), 3);
    }

    /// Only a walk can be filled.
    #[test]
    #[should_panic(expected = "only a walk's regions are staged")]
    fn a_level_that_distributes_its_tiles_cannot_be_filled_by_anyone() {
        Levels::leaf(&[(M, 64)])
            .planes(&[(M, 2)])
            .filled_by(1)
            .build();
    }

    /// A scope's instances take one level's tiles, so a second level over the same scope is
    /// refused.
    #[test]
    #[should_panic(expected = "two levels distribute over Plane")]
    fn two_levels_over_one_scope_are_refused() {
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64)]),
            Levels::leaf(&[(M, 8), (N, 8)])
                .planes(&[(M, 2)])
                .planes(&[(N, 2)])
                .build(),
        );
    }

    /// The overhanging axes are the ones some level's tile fails to divide, asked axis by axis,
    /// in the space's order: an edge at the cube level, one only a plane's box meets, and none.
    #[test]
    fn the_overhanging_axes_are_the_ones_some_level_overhangs() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 200), (N, 256), (K, 96)]),
            Levels::leaf(&[(M, 64), (N, 48), (K, 32)])
                .planes(&[(M, 2), (N, 2)])
                .walk_every(&[K])
                .cubes(&[M, N])
                .build(),
        );
        let asked: Vec<Axis> = [M, N, K]
            .into_iter()
            .filter(|&axis| partitioning.overhangs(axis))
            .collect();
        assert_eq!(partitioning.overhanging(), asked);
        assert_eq!(asked, vec![M, N]);
        assert!(staged(0).overhanging().is_empty());
    }

    /// A scope's box is the space cut down through its level, whatever walks inside it.
    #[test]
    fn a_scope_covers_the_box_its_level_cuts() {
        let partitioning = staged(0);
        let cube = Space::new(&[(M, 128), (N, 128), (K, 512)]);
        assert_eq!(partitioning.box_of(ComputeScope::Cube), cube);
        let plane = Space::new(&[(M, 64), (N, 64), (K, 64)]);
        assert_eq!(partitioning.box_of(ComputeScope::Plane), plane);
    }
}
