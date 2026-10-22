//! What the hardware instances hold of a tile's cells once a level has been distributed.

use cubecl::prelude::*;

use crate::{Axis, Carrier, ComputeScope, Count, Coverage, Level, Monoid, Space};

/// What the plane's units each hold of a tile's cells once a units level is distributed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnitShare {
    /// Every unit repeats the same work over the same whole cells.
    Repeated,
    /// Each unit carries cells of its own and reduces nothing.
    Whole,
    /// Every unit holds a partial of the same cell.
    Plane,
    /// Groups of units share one cell each; `unit_bits` are the unit-index bits the reduced axes
    /// occupy.
    Group { unit_bits: usize },
}

impl UnitShare {
    /// What `level` makes the plane's units to the cells of an operand spanning `spanned`.
    pub(crate) fn new(level: &Level, spanned: &Space) -> UnitShare {
        if level.coverage() != Coverage::Distribute(ComputeScope::Unit) {
            return UnitShare::Repeated;
        }
        // A unit of the whole launch holds its tile's cells alone.
        if level.unit_cube().is_some() {
            return UnitShare::Whole;
        }
        // Innermost first, so `weight` is the axis's stride in the unit index.
        let (mut weight, mut unit_bits) = (1usize, 0usize);
        for axis in level.axes().into_iter().rev() {
            if let Count::Distributed(_) = level.cut(axis).count {
                match spanned.contains(axis) {
                    true => continue,
                    false => return UnitShare::Plane,
                }
            }
            let plane_units = level
                .cut(axis)
                .count
                .stated()
                .expect("Level::new: a units level carries stated counts");
            if plane_units == 1 {
                continue;
            }
            assert!(
                plane_units.is_power_of_two(),
                "UnitShare: {axis:?} rides {plane_units} units, which is not a power of two, so its \
                 partials are not a bit range"
            );
            if !spanned.contains(axis) {
                unit_bits |= (plane_units - 1) * weight;
            }
            weight *= plane_units;
        }
        match unit_bits {
            0 if weight == 1 => UnitShare::Repeated,
            0 => UnitShare::Whole,
            mask if mask == weight - 1 => UnitShare::Plane,
            unit_bits => UnitShare::Group { unit_bits },
        }
    }

    /// This share under `parent`'s.
    pub(crate) fn under(self, parent: UnitShare) -> UnitShare {
        match (parent, self) {
            (UnitShare::Repeated, share) | (share, UnitShare::Repeated) => share,
            (UnitShare::Whole, share) | (share, UnitShare::Whole) => share,
            (UnitShare::Group { unit_bits: a }, UnitShare::Group { unit_bits: b }) => {
                UnitShare::Group { unit_bits: a | b }
            }
            _ => {
                panic!("UnitShare::under: {self:?} under {parent:?}: nothing reduces under a plane")
            }
        }
    }

    /// Whether a cell's partials sit on several units, so a drain has to reduce.
    pub(crate) fn reduces(self) -> bool {
        matches!(self, UnitShare::Plane | UnitShare::Group { .. })
    }

    /// The units over which `axis` of a units level is distributed, as a share.
    pub(crate) fn of_units(plane_units: usize, plane: usize) -> UnitShare {
        match plane_units {
            1 => UnitShare::Repeated,
            n if n == plane => UnitShare::Plane,
            n => {
                assert!(
                    n.is_power_of_two() && n < plane,
                    "UnitShare::of_units: a group of {n} units in a plane of {plane} is not a bit \
                     range of the unit index"
                );
                UnitShare::Group { unit_bits: n - 1 }
            }
        }
    }
}

#[cube]
impl UnitShare {
    /// Combine the partials of one cell under `monoid`, leaving the total on every holder.
    pub(crate) fn reduce_of<T: Carrier + CubePrimitive<Scalar: PlaneNumeric>>(
        value: T,
        #[comptime] share: UnitShare,
        #[comptime] monoid: Monoid,
    ) -> T {
        match comptime!(share) {
            UnitShare::Repeated | UnitShare::Whole => value,
            UnitShare::Plane => match comptime!(monoid) {
                Monoid::Sum => plane_sum(value),
                Monoid::Prod => plane_prod(value),
                Monoid::Max => plane_max(value),
                Monoid::Min => plane_min(value),
            },
            UnitShare::Group { unit_bits } => {
                let mut total = value;
                #[unroll]
                for bit in 0..comptime!(usize::BITS - unit_bits.leading_zeros()) {
                    if comptime!(unit_bits & (1 << bit) != 0) {
                        total = monoid
                            .combine::<T>(total, plane_shuffle_xor(total, comptime!(1u32 << bit)));
                    }
                }
                total
            }
        }
    }
}

impl UnitShare {
    /// Combine the partials of one cell under `monoid`, leaving the total on every holder.
    pub fn reduce<T: Carrier + CubePrimitive<Scalar: PlaneNumeric>>(
        self,
        value: T,
        monoid: Monoid,
    ) -> T {
        UnitShare::reduce_of::<T>(value, self, monoid)
    }

    pub fn __expand_reduce_method<T: Carrier + CubePrimitive<Scalar: PlaneNumeric>>(
        self,
        scope: &Scope,
        value: T::ExpandType,
        monoid: Monoid,
    ) -> T::ExpandType {
        UnitShare::__expand_reduce_of::<T>(scope, value, self, monoid)
    }
}

/// What one plane or cube instance holds of a tile's cells, and across which instances the
/// partials of one cell lie: what a destination that adds has to serialize.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum SplitShare {
    /// Every cell this instance writes is its own.
    Whole,
    /// Several cubes hold partials of the same cell, each writing it whole from its planes.
    PartialAcrossCubes,
    /// Several planes of one cube hold partials of the same cell, and write it at once.
    PartialAcrossPlanes,
}

impl SplitShare {
    /// What one instance of an operand spanning `spanned` holds after `level` is distributed over
    /// `space`. `space` must be the level's whole space, not the operand's projection.
    pub(crate) fn new(level: &Level, space: &Space, spanned: &Space) -> SplitShare {
        match level.coverage() {
            Coverage::Walk | Coverage::Distribute(ComputeScope::Unit) => return SplitShare::Whole,
            Coverage::Distribute(ComputeScope::Plane)
            | Coverage::Distribute(ComputeScope::PlaneGroup { .. })
            | Coverage::Distribute(ComputeScope::Cube) => {}
        }
        // A shared flat index over an unspanned axis may cover part of a cell.
        let split = match level.shared_by() {
            Some(_) => level.axes().iter().any(|axis| !spanned.contains(*axis)),
            None => level.axes().into_iter().any(|axis: Axis| {
                !spanned.contains(axis) && level.instances_along(space, axis) != Some(1)
            }),
        };
        match (split, level.coverage()) {
            (false, _) => SplitShare::Whole,
            (true, Coverage::Distribute(ComputeScope::Plane | ComputeScope::PlaneGroup { .. })) => {
                SplitShare::PartialAcrossPlanes
            }
            (true, _) => SplitShare::PartialAcrossCubes,
        }
    }

    /// This share under `parent`'s: partial stays partial, and a split across planes, which
    /// writes a cell from several places at once, outranks one across cubes.
    pub(crate) fn under(self, parent: SplitShare) -> SplitShare {
        match (parent, self) {
            (SplitShare::PartialAcrossPlanes, _) | (_, SplitShare::PartialAcrossPlanes) => {
                SplitShare::PartialAcrossPlanes
            }
            (SplitShare::PartialAcrossCubes, _) | (_, SplitShare::PartialAcrossCubes) => {
                SplitShare::PartialAcrossCubes
            }
            (SplitShare::Whole, SplitShare::Whole) => SplitShare::Whole,
        }
    }
}

/// A count a units level states along an axis, as the units it takes.
impl From<Count> for usize {
    fn from(count: Count) -> usize {
        count
            .stated()
            .expect("a units level's count is stated; every tile is the launch's")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Levels, Space};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A cube-cut contraction is partial to the output.
    #[test]
    fn a_cube_cut_contraction_is_partial_to_the_output() {
        let space = Space::new(&[(M, 4), (N, 4), (K, 8)]);
        let level = Levels::leaf(&[(K, 4)]).cubes(&[K]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::PartialAcrossCubes
        );
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, K])),
            SplitShare::Whole
        );
    }

    /// Same at plane scope.
    #[test]
    fn a_plane_cut_contraction_is_partial_to_the_output() {
        let space = Space::new(&[(M, 4), (N, 4), (K, 8)]);
        let level = Levels::leaf(&[(K, 4)]).planes(&[(K, 2)]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::PartialAcrossPlanes
        );
    }

    /// A shared grid over the contraction is partial to the output.
    #[test]
    fn a_shared_grid_is_partial_to_the_output() {
        let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
        let level = Levels::leaf(&[(M, 4), (N, 4), (K, 8)])
            .cubes(&[M, N, K])
            .shared_by(3)
            .level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::PartialAcrossCubes
        );
        assert_eq!(SplitShare::new(&level, &space, &space), SplitShare::Whole);
    }

    /// A cube cut of the whole axis is not a split.
    #[test]
    fn a_cube_cut_of_the_whole_axis_is_not_a_split() {
        let space = Space::new(&[(M, 4), (N, 4), (K, 8)]);
        let level = Levels::leaf(&[(N, 1), (K, 8)]).cubes(&[N, K]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::Whole
        );
    }

    /// A cube cut of an output axis stays whole.
    #[test]
    fn a_cube_cut_output_axis_stays_whole() {
        let space = Space::new(&[(M, 4), (N, 8), (K, 4)]);
        let level = Levels::leaf(&[(N, 4)]).cubes(&[N]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::Whole
        );
    }

    /// A units level reduces what the operand does not span.
    #[test]
    fn a_units_level_reduces_what_the_operand_does_not_span() {
        let space = Space::new(&[(M, 4), (N, 8), (K, 32)]);
        let out = space.subspace(&[M, N]);
        let plane = Levels::leaf(&[(K, 1)]).units(&[(K, 32)]).level();
        assert_eq!(UnitShare::new(&plane, &out), UnitShare::Plane);
        let whole = Levels::leaf(&[(N, 1)]).units(&[(N, 8)]).level();
        assert_eq!(UnitShare::new(&whole, &out), UnitShare::Whole);
        let team = Levels::leaf(&[(N, 1), (K, 1)])
            .units(&[(N, 8), (K, 4)])
            .level();
        assert_eq!(
            UnitShare::new(&team, &out),
            UnitShare::Group { unit_bits: 3 }
        );
        let walk = Levels::leaf(&[(K, 8)]).walk_every(&[K]).level();
        assert_eq!(UnitShare::new(&walk, &out), UnitShare::Repeated);
    }

    /// Shares compose level by level.
    #[test]
    fn shares_compose_level_by_level() {
        let group = UnitShare::Group { unit_bits: 3 };
        assert_eq!(group.under(UnitShare::Repeated), group);
        assert_eq!(UnitShare::Whole.under(group), group);
        assert_eq!(
            UnitShare::Group { unit_bits: 12 }.under(group),
            UnitShare::Group { unit_bits: 15 }
        );
        assert_eq!(
            UnitShare::Repeated.under(UnitShare::Whole),
            UnitShare::Whole
        );
        assert_eq!(
            SplitShare::Whole.under(SplitShare::PartialAcrossCubes),
            SplitShare::PartialAcrossCubes
        );
        // A split across planes under one across cubes still has planes writing a cell at once.
        assert_eq!(
            SplitShare::PartialAcrossPlanes.under(SplitShare::PartialAcrossCubes),
            SplitShare::PartialAcrossPlanes
        );
        assert_eq!(
            SplitShare::PartialAcrossCubes.under(SplitShare::PartialAcrossPlanes),
            SplitShare::PartialAcrossPlanes
        );
    }
}
