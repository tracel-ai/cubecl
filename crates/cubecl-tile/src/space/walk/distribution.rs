//! How one axis of a walk's space is distributed at its level, and where an instance finds its
//! own position.

use cubecl::prelude::*;

use crate::{
    Axis, ComputeScope, Count, CubeAxis, Extent, Integer, IntegerExpand, Level, Space, Spread,
};

/// One axis of a level: walked whole by every instance, or distributed over the level's scope.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum AxisDistribution {
    /// Every instance steps the whole grid along this axis.
    Walked,
    /// The grid is distributed over the scope's instances ([`Distribution`]).
    Distributed(Distribution),
}

/// How an axis's grid is distributed over a level's scope.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Distribution {
    /// The scope whose instances take the tiles.
    pub(crate) scope: ComputeScope,
    /// The grid dimension a cube level's axis rides.
    pub(crate) dim: Option<CubeAxis>,
    pub(crate) spread: Spread,
    /// The workers an [`AllAcross`](Count::AllAcross) axis is distributed to in runs.
    pub(crate) across: Option<usize>,
    /// Whether the tiles are taken in turns by as many units as the launch runs.
    pub(crate) in_turns: bool,
    /// Whether the host proved every worker's run whole, so the kernel skips clamping it.
    pub(crate) divides: bool,
    /// The later axes sharing this axis's hardware dimension.
    pub(crate) inner: Vec<usize>,
    /// The instance weight of same-dimension axes inside this one that the space does not span.
    pub(crate) unspanned: usize,
    /// Which of the two jointly decoded in-plane positions this axis takes, when swizzled.
    pub(crate) swizzled: Option<usize>,
}

impl AxisDistribution {
    /// How `level` distributes axis `p` of `space`.
    pub(crate) fn new(
        level: &Level,
        space: &Space,
        p: usize,
        swizzled: Option<(usize, usize)>,
    ) -> AxisDistribution {
        let axis = space.axis_at(p);
        if !level.distributes(axis) {
            return AxisDistribution::Walked;
        }
        let dim = level.cube_axis(axis);
        let same_dim = |q: usize| {
            let other = space.axis_at(q);
            level.distributes(other) && level.cube_axis(other) == dim
        };
        AxisDistribution::Distributed(Distribution {
            scope: level
                .coverage()
                .scope()
                .expect("a level that distributes an axis distributes it over a scope"),
            dim,
            spread: level.cut(axis).spread,
            across: match level.cut(axis).count {
                Count::AllAcross(workers) => Some(workers),
                Count::Stated(_) | Count::All | Count::Distributed(_) => None,
            },
            in_turns: matches!(level.cut(axis).count, Count::Distributed(_)),
            divides: AxisDistribution::divides(level, space, axis),
            inner: ((p + 1)..space.rank()).filter(|&q| same_dim(q)).collect(),
            unspanned: AxisDistribution::unspanned_weight(level, space, axis),
            swizzled: match swizzled {
                Some((x, _)) if x == p => Some(0),
                Some((_, y)) if y == p => Some(1),
                _ => None,
            },
        })
    }

    /// Whether every worker's run along an axis is the full one.
    fn divides(level: &Level, space: &Space, axis: Axis) -> bool {
        match level.cut(axis).count {
            Count::AllAcross(workers) => match space.extent_raw(axis) {
                Extent::Static(extent) => extent
                    .div_ceil(level.cut(axis).tile)
                    .is_multiple_of(workers),
                Extent::Dynamic => false,
            },
            Count::Distributed(_) => false,
            Count::Stated(_) | Count::All => true,
        }
    }

    /// The instance counts of same-dimension axes inside `axis` that `space` does not span.
    /// Panics where such an axis has no comptime count.
    pub(crate) fn unspanned_weight(level: &Level, space: &Space, axis: Axis) -> usize {
        let dim = level.cube_axis(axis);
        level
            .axes()
            .iter()
            .skip_while(|&&a| a != axis)
            .skip(1)
            .filter(|&&a| !space.contains(a) && level.distributes(a) && level.cube_axis(a) == dim)
            .map(|&a| {
                level.cut(a).count.stated().unwrap_or_else(|| {
                    panic!(
                        "AxisDistribution: {a:?} is distributed inside {axis:?} over the same scope but this \
                         operand does not span it, and its instance count is not comptime, so \
                         {axis:?}'s digit of the instance index cannot be decoded"
                    )
                })
            })
            .product()
    }
}

#[cube]
impl AxisDistribution {
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

    /// The raw hardware position of the instances this distribution hands tiles to.
    pub(crate) fn hardware(
        #[comptime] compute_scope: ComputeScope,
        #[comptime] dim: Option<CubeAxis>,
    ) -> usize {
        match comptime!(compute_scope) {
            ComputeScope::Cube => CubeAxis::position(comptime!(
                dim.expect("a cube level's distributed axis rides a grid dimension")
            )),
            ComputeScope::Plane | ComputeScope::PlaneGroup { .. } | ComputeScope::Unit => {
                ComputeScope::position(compute_scope)
            }
        }
    }
}
