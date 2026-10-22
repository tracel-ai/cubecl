//! Where a loop is: the box a kernel's loops reached ([`Region`]), and the leaves and planes below
//! it.

use super::{Level, Partitioning, Space};
use crate::{
    Axis, ComputeScope, Coords, Coverage, Integer, IntegerExpand, MatrixAxes, Path, Step, Walk,
};
use cubecl::{prelude::*, unexpanded};

/// The path a kernel's loops took to a box, with the coordinates each loop handed out.
/// Iterating distributes the next level.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Region {
    digits: Sequence<Coords<u32>>,
    /// The runtime sizes of the root's dynamic axes, positional.
    sizes: Sequence<usize>,
    #[cube(comptime)]
    pub(crate) path: Path,
}

#[cube]
impl Region {
    pub(crate) fn new(
        digits: Sequence<Coords<u32>>,
        sizes: Sequence<usize>,
        #[comptime] path: Path,
    ) -> Region {
        Region {
            digits,
            sizes,
            path,
        }
    }

    /// The empty path at the partitioning's space.
    pub(crate) fn root(partitioning: &Partitioning) -> Region {
        Region::rooted(
            &partitioning.space,
            comptime!(partitioning.levels.clone()),
            0usize,
        )
    }

    /// The empty path at `space`, which sits at `depth` in a partitioning of `levels`.
    pub(crate) fn rooted(
        space: &Space,
        #[comptime] levels: Vec<Level>,
        #[comptime] depth: usize,
    ) -> Region {
        Region::new(
            Sequence::new(),
            space.sizes.clone(),
            comptime!(Path::new(
                depth,
                Partitioning::new(space.clone(), levels),
                Vec::new()
            )),
        )
    }

    /// The regions of the level below this one.
    pub fn walk(&self) -> Walk {
        Walk::of(&self.child(), comptime!(self.path.next()), self.clone())
    }

    /// The regions at the partitioning's innermost level below this one, one loop per level
    /// between: a kernel reads the same however many levels its partitioning stacks there.
    pub fn leaves(&self) -> Leaves {
        Leaves {
            region: self.clone(),
        }
    }

    /// The planes below this region, each once, at the first step of every walk between: what a
    /// plane holds across a walk, such as its rows, is the same at every step, while its
    /// coordinates along a walked axis are the first step's.
    pub fn planes(&self) -> Planes {
        Planes {
            region: self.clone(),
        }
    }

    /// The region one level below the root at trailing-two coordinates `(c0, c1)` under `level`,
    /// `0` elsewhere, for a tile at `depth`. Comptime coordinates stay constant.
    pub fn trailing(
        #[comptime] depth: usize,
        #[comptime] space: Space,
        #[comptime] level: Level,
        c0: usize,
        c1: usize,
    ) -> Region {
        let edges = comptime!(MatrixAxes::edges(&space));
        Region::cell(depth, space, level, edges, c0, c1)
    }

    /// The region one level below the root at coordinates `(c0, c1)` along the first axis of
    /// each of `edges`' two groups under `level`, `0` elsewhere, for a tile at `depth`.
    pub(crate) fn cell(
        #[comptime] depth: usize,
        #[comptime] space: Space,
        #[comptime] level: Level,
        #[comptime] edges: MatrixAxes,
        c0: usize,
        c1: usize,
    ) -> Region {
        let rank = comptime!(space.rank());
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            // `cast`, not `as`, and `runtime` on the `0`: both keep a comptime coordinate constant.
            let c = if comptime!(p == edges.row_split) {
                c0.cast::<u32>()
            } else if comptime!(p == edges.col_split) {
                c1.cast::<u32>()
            } else {
                0u32.runtime()
            };
            coords.push(c);
        }
        let mut digits = Sequence::<Coords<u32>>::new();
        digits.push(coords);
        Region::new(
            digits,
            Sequence::new(),
            comptime!(Path::new(
                depth,
                Partitioning::new(space.clone(), Vec::new()),
                vec![level]
            )),
        )
    }

    /// The `i`-th step of the path.
    pub(crate) fn step(&self, #[comptime] i: usize) -> Step {
        Step::new(
            self.digits.index(i).clone(),
            comptime!(self.path.space_at(i)),
            comptime!(self.path.level(i).clone()),
            comptime!(self.path.depth() - self.path.len() + i),
        )
    }

    /// The box this region covers, as a runtime space.
    pub(crate) fn child(&self) -> Space {
        Space::with_sizes(comptime!(self.path.child()), self.sizes.clone())
    }

    /// The innermost level's coordinate along `axis`; `0` when the axis is absent.
    pub fn coord(&self, #[comptime] axis: Axis) -> usize {
        let last = comptime!(self.path.len() - 1);
        self.step(last).coord(axis)
    }

    /// The first coordinate along `axis` of the box this region covers, counted from its root's:
    /// every level's coordinate along `axis` times the tile it cuts there. What a kernel bounds a
    /// walk of its own by, as an attention's causal queries bound the keys they read.
    pub fn origin(&self, #[comptime] axis: Axis) -> usize {
        let mut origin = 0usize;
        #[unroll]
        for i in 0..comptime!(self.path.len()) {
            if let Some(tile) = comptime!(self.path.level(i).tile(axis)) {
                origin += self.step(i).coord(axis) * tile;
            }
        }
        origin
    }

    /// This path with one more level's coordinates below it.
    pub(crate) fn below(&self, coords: Coords<u32>, #[comptime] level: Level) -> Region {
        let n = comptime!(self.digits.len());
        let mut digits = Sequence::<Coords<u32>>::new();
        #[unroll]
        for i in 0..n {
            digits.push(self.digits.index(i).clone());
        }
        digits.push(coords);
        Region::new(
            digits,
            self.sizes.clone(),
            comptime!(self.path.below(level)),
        )
    }

    /// The regions of `level` over this region's own box.
    pub fn over(&self, #[comptime] level: &Level) -> Walk {
        Walk::of(&self.child(), comptime!(level.clone()), self.clone())
    }
}

/// Host-side stand-in for `for plane in cube`; never runs.
impl IntoIterator for Region {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

impl IntoIterator for &Region {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// `for plane in cube` iterates the level below `cube`.
impl Iterable for RegionExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let walk = self.__expand_walk_method(scope);
        if walk.const_len() == Some(1) {
            walk.expand_unroll(scope, body)
        } else {
            walk.expand(scope, body)
        }
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.__expand_walk_method(scope).expand_unroll(scope, body)
    }
}

/// `for plane in &cube`: the same, leaving `cube` to the body.
impl Iterable for &RegionExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand_unroll(scope, body)
    }
}

/// The regions at a partitioning's innermost level below one region ([`Region::leaves`]).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Leaves {
    #[allow(dead_code)] // Read at expansion, through `LeavesExpand`.
    region: Region,
}

/// Host-side stand-in for `for unit in plane.leaves()`; never runs.
impl IntoIterator for Leaves {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// One loop per level between the region and the partitioning's innermost level.
impl Iterable for LeavesExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        each_leaf(scope, self.region, body, false);
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        each_leaf(scope, self.region, body, true);
    }
}

/// The planes below one region, each once ([`Region::planes`]).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Planes {
    #[allow(dead_code)] // Read at expansion, through `PlanesExpand`.
    region: Region,
}

/// Host-side stand-in for `for plane in cube.planes()`; never runs.
impl IntoIterator for Planes {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// A walk's first step per walk level, then a loop over the plane level.
impl Iterable for PlanesExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        each_plane(scope, self.region, body, false);
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        each_plane(scope, self.region, body, true);
    }
}

/// `body` over the planes below `region`. A plane's place is the same at every step of a walk, so
/// each walk between is read at its first step.
fn each_plane(
    scope: &Scope,
    region: RegionExpand,
    body: &mut dyn FnMut(&Scope, RegionExpand),
    unroll: bool,
) {
    match region.path.next().coverage() {
        Coverage::Distribute(ComputeScope::Plane) => match unroll {
            true => region.expand_unroll(scope, body),
            false => region.expand(scope, body),
        },
        Coverage::Walk => {
            let first = region
                .__expand_walk_method(scope)
                .__expand_region_method(scope, 0usize.into_expand(scope));
            each_plane(scope, first, body, unroll)
        }
        Coverage::Distribute(other) => panic!(
            "Region::planes: a level dealt to each {other:?} lies below this region before any \
             plane level, so this region has no planes of its own"
        ),
    }
}

/// `body` over every region at the bottom of `region`'s partitioning below it.
fn each_leaf(
    scope: &Scope,
    region: RegionExpand,
    body: &mut dyn FnMut(&Scope, RegionExpand),
    unroll: bool,
) {
    if region.path.at_bottom() {
        body(scope, region);
        return;
    }
    let mut below = |scope: &Scope, child: RegionExpand| each_leaf(scope, child, body, unroll);
    match unroll {
        true => region.expand_unroll(scope, &mut below),
        false => region.expand(scope, &mut below),
    }
}
