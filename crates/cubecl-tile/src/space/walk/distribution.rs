//! How one dimension of a walk's space is distributed at its level, and where an instance finds its
//! own position.

use cubecl::prelude::*;

use crate::{
    ComputeScope, Count, CubeAxis, Dimension, Extent, Integer, IntegerExpand, Level, Space, Spread,
};

/// One dimension of a level: walked whole by every instance, or distributed over the level's scope.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum DimensionDistribution {
    /// Every instance steps the whole grid along this dimension.
    Walked,
    /// The grid is distributed over the scope's instances ([`Distribution`]).
    Distributed(Distribution),
}

/// How a dimension's grid is distributed over a level's scope.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Distribution {
    /// The scope whose instances take the tiles.
    pub(crate) scope: ComputeScope,
    /// The grid axis a cube level's dimension rides.
    pub(crate) cube_axis: Option<CubeAxis>,
    pub(crate) spread: Spread,
    /// Who the dimension's tiles are dealt to.
    pub(crate) workers: Workers,
    /// Whether the host proved every worker's run whole, so the kernel skips clamping it.
    pub(crate) divides: bool,
    /// The later dimensions sharing this dimension's hardware dimension.
    pub(crate) inner: Vec<usize>,
    /// The instance weight of same-dimension dimensions inside this one that the space does not
    /// span.
    pub(crate) unspanned: usize,
    /// Which of the two jointly decoded in-plane positions this dimension takes, when swizzled.
    pub(crate) swizzled: Option<usize>,
}

/// Who a distributed dimension's tiles are dealt to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Workers {
    /// One instance of the scope per tile of the grid.
    Grid,
    /// This many workers, each taking a run of tiles ([`AllAcross`](Count::AllAcross)).
    Across(usize),
    /// The units of one instance of `within`, taking the tiles in turns
    /// ([`InTurns`](Count::InTurns)), `units` of them where the launch states it, the cube's last
    /// `fillers` planes' alone where it sets them aside.
    Units {
        within: ComputeScope,
        units: Option<usize>,
        fillers: usize,
    },
}

impl DimensionDistribution {
    /// How `level` distributes dimension `p` of `space`.
    pub(crate) fn new(
        level: &Level,
        space: &Space,
        p: usize,
        swizzled: Option<(usize, usize)>,
    ) -> DimensionDistribution {
        let dimension = space.dimension_at(p);
        if !level.distributes(dimension) {
            return DimensionDistribution::Walked;
        }
        let cube_axis = level.cube_axis(dimension);
        let same_dim = |q: usize| {
            let other = space.dimension_at(q);
            level.distributes(other) && level.cube_axis(other) == cube_axis
        };
        DimensionDistribution::Distributed(Distribution {
            scope: level
                .coverage()
                .scope()
                .expect("a level that distributes a dimension should distribute it over a scope"),
            cube_axis,
            spread: level.cut(dimension).spread,
            workers: match level.cut(dimension).count {
                Count::AllAcross(workers) => Workers::Across(workers),
                Count::InTurns { within, units, .. } => Workers::Units {
                    within,
                    units,
                    fillers: level.fillers(),
                },
                Count::Stated(_) | Count::All => Workers::Grid,
            },
            divides: DimensionDistribution::divides(level, space, dimension),
            inner: ((p + 1)..space.rank()).filter(|&q| same_dim(q)).collect(),
            unspanned: DimensionDistribution::unspanned_weight(level, space, dimension),
            swizzled: match swizzled {
                Some((x, _)) if x == p => Some(0),
                Some((_, y)) if y == p => Some(1),
                _ => None,
            },
        })
    }

    /// Whether every worker's run along a dimension is the full one.
    fn divides(level: &Level, space: &Space, dimension: Dimension) -> bool {
        match level.cut(dimension).count {
            Count::AllAcross(workers) => match space.extent_raw(dimension) {
                Extent::Static(extent) => extent
                    .div_ceil(level.cut(dimension).tile)
                    .is_multiple_of(workers),
                Extent::Dynamic => false,
            },
            Count::InTurns { units, .. } => match (level.tiles_const(space, dimension), units) {
                (Some(tiles), Some(units)) => tiles.is_multiple_of(units),
                _ => false,
            },
            Count::Stated(_) | Count::All => true,
        }
    }

    /// The instance counts of same-dimension dimensions inside `dimension` that `space` does not
    /// span. Panics where such a dimension has no comptime count.
    pub(crate) fn unspanned_weight(level: &Level, space: &Space, dimension: Dimension) -> usize {
        let cube_axis = level.cube_axis(dimension);
        level
            .dimensions()
            .iter()
            .skip_while(|&&a| a != dimension)
            .skip(1)
            .filter(|&&a| !space.contains(a) && level.distributes(a) && level.cube_axis(a) == cube_axis)
            .map(|&a| {
                level.cut(a).count.stated().unwrap_or_else(|| {
                    panic!(
                        "DimensionDistribution: {a:?} is distributed inside {dimension:?} over the same scope but this \
                         operand does not span it, and its instance count is not comptime, so \
                         {dimension:?}'s coordinate in the instance index cannot be decoded"
                    )
                })
            })
            .product()
    }
}

#[cube]
impl DimensionDistribution {
    /// The tiles instance `pos` of `instances` takes of a `grid` distributed in runs of `run`.
    pub(crate) fn tiles(
        grid: usize,
        pos: usize,
        instances: usize,
        run: usize,
        #[comptime] spread: Spread,
        #[comptime] divides: bool,
    ) -> usize {
        if divides {
            run
        } else {
            match spread {
                Spread::Contiguous => {
                    let start = pos.times(run);
                    run.min_with(grid.max_with(start).minus(start))
                }
                Spread::Interleaved => grid
                    .max_with(pos)
                    .minus(pos)
                    .plus(instances.minus(1usize))
                    .divided_by(instances),
            }
        }
    }

    /// How many units of one instance of `within` take a level's tiles in turns: `units` where the
    /// launch states them, read off the launch otherwise, the cube's last `fillers` planes' alone
    /// where it sets them aside.
    pub(crate) fn units_within(
        #[comptime] within: ComputeScope,
        #[comptime] units: Option<usize>,
        #[comptime] fillers: usize,
    ) -> usize {
        match (units, fillers) {
            (Some(units), _) => units.runtime(),
            (None, 0) => match within {
                ComputeScope::Cube => CUBE_DIM as usize,
                ComputeScope::PlaneGroup { planes } => planes * CUBE_DIM_X as usize,
                ComputeScope::Plane => CUBE_DIM_X as usize,
                ComputeScope::Unit => 1usize.runtime(),
            },
            (None, fillers) => fillers * CUBE_DIM_X as usize,
        }
    }

    /// This unit's place among the units [`units_within`](DimensionDistribution::units_within)
    /// counts: its place in its instance of `within` ([`ComputeScope::unit`]), among the filling
    /// planes' units where the cube sets the last `fillers` aside.
    pub(crate) fn unit_within(
        #[comptime] within: ComputeScope,
        #[comptime] fillers: usize,
    ) -> usize {
        match fillers {
            0 => ComputeScope::unit(within),
            fillers => (UNIT_POS - (CUBE_DIM - comptime!(fillers as u32) * CUBE_DIM_X)) as usize,
        }
    }

    /// The raw hardware position of the instances this distribution hands tiles to.
    pub(crate) fn hardware(
        #[comptime] compute_scope: ComputeScope,
        #[comptime] cube_axis: Option<CubeAxis>,
    ) -> usize {
        match compute_scope {
            ComputeScope::Cube => CubeAxis::position(
                cube_axis.expect("a cube level's distributed dimension should ride a grid axis"),
            ),
            ComputeScope::Plane | ComputeScope::PlaneGroup { .. } | ComputeScope::Unit => {
                ComputeScope::position(compute_scope)
            }
        }
    }
}
