//! The comptime shape of a region's path: the levels taken from a root space ([`Path`]).

use super::{ComputeScope, Level, Partitioning, Space};
use crate::Axis;

/// The comptime shape of a [`Region`](crate::Region)'s path: the levels taken from a root space,
/// outermost first.
#[derive(Clone, Debug)]
pub(crate) struct Path {
    base: usize,
    root: Partitioning,
    levels: Vec<Level>,
}

impl Path {
    pub(crate) fn new(base: usize, root: Partitioning, levels: Vec<Level>) -> Self {
        Path { base, root, levels }
    }

    /// The depth of the box this path names.
    pub(crate) fn depth(&self) -> usize {
        self.base + self.levels.len()
    }

    pub(crate) fn base(&self) -> usize {
        self.base
    }

    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn level(&self, i: usize) -> &Level {
        &self.levels[i]
    }

    /// The space the `i`-th level of the path cuts.
    pub(crate) fn space_at(&self, i: usize) -> Space {
        self.root.space().leaf(&self.levels[..i])
    }

    /// The box this path names.
    pub(crate) fn child(&self) -> Space {
        self.root.space().leaf(&self.levels)
    }

    /// The partitioning's level at this path's depth.
    pub(crate) fn next(&self) -> Level {
        let depth = self.depth();
        let levels = self.root.levels();
        assert!(
            depth < levels.len(),
            "this region sits {depth} levels down a partitioning of {} levels, so there is no \
             level below it to iterate; a walk of the kernel's own says `over`",
            levels.len()
        );
        levels[depth].clone()
    }

    /// Who runs the box this path names: the narrowest scope any level from the root down to it
    /// hands its tiles to — a unit, a plane, or, where every level is walked or distributed over
    /// cubes, the cube.
    pub(crate) fn scope(&self) -> ComputeScope {
        self.root.levels()[..self.base]
            .iter()
            .chain(&self.levels)
            .filter_map(|level| level.coverage().scope())
            .min()
            .unwrap_or(ComputeScope::Cube)
    }

    /// The root partitioning's levels, outermost first: what a tile placed on this path is walked
    /// with.
    pub(crate) fn root_levels(&self) -> Vec<Level> {
        self.root.levels().to_vec()
    }

    /// The planes one cube of the root's partitioning holds ([`Partitioning::planes_per_cube`]).
    pub(crate) fn planes_per_cube(&self) -> usize {
        self.root.planes_per_cube() as usize
    }

    /// The axes the root's partitioning distributes across a cube's planes that `along` admits,
    /// each with its count of planes, outermost first: the planes holding the same box of a tile
    /// that lacks those axes, or splitting the same slices of one that runs along them.
    pub(crate) fn planes_along(&self, along: impl Fn(Axis) -> bool) -> Vec<(Axis, usize)> {
        let mut handed = self.root.space().clone();
        let mut planes = Vec::new();
        for level in self.root.levels() {
            let scope = level.coverage().scope();
            for axis in level.axes() {
                if !(scope.and_then(ComputeScope::planes).is_some()
                    && level.distributes(axis)
                    && along(axis))
                {
                    continue;
                }
                assert!(
                    scope == Some(ComputeScope::Plane),
                    "{axis:?} is split across the cube's plane groups, and nothing merges what \
                     plane groups hold of the same box yet; deal each group a box of its own"
                );
                let count = level.instances_along(&handed, axis).unwrap_or_else(|| {
                    panic!(
                        "{axis:?} is split across the cube's planes at a count only the launch \
                         knows, so the planes holding the same box cannot be counted"
                    )
                });
                planes.push((axis, count));
            }
            handed = level.child(&handed);
        }
        planes
    }

    /// Whether this path reaches the partitioning's innermost level, with no level below it.
    pub(crate) fn at_bottom(&self) -> bool {
        self.depth() == self.root.levels().len()
    }

    /// This path one level further down.
    pub(crate) fn below(&self, level: Level) -> Path {
        let mut levels = self.levels.clone();
        levels.push(level);
        Path::new(self.base, self.root.clone(), levels)
    }
}
