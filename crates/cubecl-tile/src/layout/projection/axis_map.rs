//! One physical axis of a buffer as an affine combination of logical axes, and its coefficients.

use cubecl::zspace::SmallVec;

use crate::space::extent::Extent;
use crate::{Axis, Space, algebra::gcd};

/// How far one unit of a logical axis moves along one physical axis.
/// A `Dynamic` coefficient declares `max`, the largest value the launch may pass.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scale {
    Static(usize),
    Dynamic { max: usize },
}

impl Scale {
    /// The comptime coefficient; panics on `Dynamic`.
    pub fn get(self) -> usize {
        match self {
            Scale::Static(n) => n,
            Scale::Dynamic { .. } => {
                panic!(
                    "Scale::get: this coefficient is Dynamic; its value is only known at runtime"
                )
            }
        }
    }

    /// The largest value this coefficient can take: itself, or the declared `max`.
    pub fn bound(self) -> usize {
        match self {
            Scale::Static(n) => n,
            Scale::Dynamic { max } => max,
        }
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Scale::Dynamic { .. })
    }
}

/// The constant term of one physical axis's affine combination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Offset {
    Static(isize),
    Dynamic,
}

impl Offset {
    /// The comptime offset; panics on `Dynamic`.
    pub fn get(self) -> isize {
        match self {
            Offset::Static(n) => n,
            Offset::Dynamic => {
                panic!("Offset::get: this offset is Dynamic; its value is only known at runtime")
            }
        }
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Offset::Dynamic)
    }
}

impl From<isize> for Offset {
    fn from(n: isize) -> Self {
        Offset::Static(n)
    }
}

/// What one physical axis's affine combination is divided by, floored.
/// A `Dynamic` divisor declares `min`, the smallest value the launch may pass.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Divisor {
    Static(usize),
    Dynamic { min: usize },
}

impl Divisor {
    pub fn is_dynamic(self) -> bool {
        matches!(self, Divisor::Dynamic { .. })
    }

    /// The smallest value this divisor can take: itself, or the declared `min`.
    pub fn bound(self) -> usize {
        match self {
            Divisor::Static(d) => d,
            Divisor::Dynamic { min } => min,
        }
    }

    /// Whether this divides by `1`; a `Dynamic` divisor never is.
    pub(crate) fn is_unit(self) -> bool {
        self == Divisor::Static(1)
    }
}

impl From<usize> for Divisor {
    fn from(n: usize) -> Self {
        Divisor::Static(n)
    }
}

/// One logical axis's contribution to one physical axis: `digit * scale`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AxisTerm {
    pub axis: Axis,
    pub scale: Scale,
}

/// Whether a physical axis's terms can land on the same cell, as stated by the caller.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Composition {
    /// No two logical positions share a cell: the terms partition the axis.
    Disjoint,
    /// Two positions may land on the same cell, as with a stencil or a resample.
    Overlapping,
}

/// One buffer dim as `(Σ digit(axis) * scale + offset) / divisor`, floored.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxisMap {
    terms: SmallVec<[AxisTerm; Space::MAX_RANK]>,
    offset: Offset,
    divisor: Divisor,
    composition: Composition,
}

/// What a physical axis is addressed by.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Addressed {
    /// The leading logical axis of this map's terms, naming its coordinate group.
    By(Axis),
    /// Nothing: every position resolves to the same element.
    Broadcast,
}

impl PhysicalAxisMap {
    /// A map addressing no logical axis: every position resolves to the same element.
    pub fn broadcast() -> Self {
        PhysicalAxisMap::scaled(&[])
    }

    pub fn of(axis: Axis) -> Self {
        PhysicalAxisMap {
            terms: SmallVec::from_slice(&[AxisTerm {
                axis,
                scale: Scale::Static(1),
            }]),
            offset: Offset::Static(0),
            divisor: Divisor::Static(1),
            composition: Composition::Disjoint,
        }
    }

    /// This physical axis partitioned by several logical ones, coarsest first:
    /// `disjoint(&[(KB, 32), (KI, 1)])` is `k = kb * 32 + ki`.
    pub fn disjoint(terms: &[(Axis, usize)]) -> Self {
        let mut map = Self::affine(terms);
        map.composition = Composition::Disjoint;
        map
    }

    /// An affine combination, e.g. `affine(&[(Oh, stride), (Rh, dilation)])`.
    pub fn affine(terms: &[(Axis, usize)]) -> Self {
        let terms: SmallVec<[(Axis, Scale); Space::MAX_RANK]> = terms
            .iter()
            .map(|&(axis, scale)| (axis, Scale::Static(scale)))
            .collect();
        Self::scaled(&terms)
    }

    /// [`affine`](Self::affine) over explicit [`Scale`]s, for runtime coefficients.
    pub fn scaled(terms: &[(Axis, Scale)]) -> Self {
        for &(axis, scale) in terms {
            assert!(
                scale.bound() > 0,
                "PhysicalAxisMap: {axis:?}'s coefficient is bounded at 0, so it addresses no cell; \
                 drop the term instead"
            );
        }
        // Only the identity is known to partition; anything else must be claimed `disjoint`.
        let composition = match terms {
            [(_, Scale::Static(1))] => Composition::Disjoint,
            _ => Composition::Overlapping,
        };
        PhysicalAxisMap {
            terms: terms
                .iter()
                .map(|&(axis, scale)| AxisTerm { axis, scale })
                .collect(),
            offset: Offset::Static(0),
            divisor: Divisor::Static(1),
            composition,
        }
    }

    /// The same combination plus a signed constant or dynamic `offset`, added before any
    /// [`over`](Self::over).
    pub fn shifted(mut self, offset: impl Into<Offset>) -> Self {
        let offset = offset.into();
        assert!(
            self.divisor.is_unit() && self.offset == Offset::Static(0),
            "PhysicalAxisMap::shifted: shift once, before `over`"
        );
        if offset != Offset::Static(0) {
            self.composition = Composition::Overlapping;
        }
        self.offset = offset;
        self
    }

    /// The same combination divided by `divisor`, floored; a divisor every coefficient cancels
    /// is reduced away.
    pub fn over(mut self, divisor: impl Into<Divisor>) -> Self {
        let divisor = divisor.into();
        assert!(
            divisor.bound() > 0,
            "PhysicalAxisMap::over: a divisor of 0 does not map anywhere"
        );
        assert!(
            self.divisor.is_unit(),
            "PhysicalAxisMap::over: this map already divides by {:?}; state the whole divisor once",
            self.divisor
        );
        self.divisor = divisor;
        self.composition = Composition::Overlapping;
        self.reduced()
    }

    /// The same mapping without a divisor, when every static coefficient is divisible by it.
    fn reduced(mut self) -> Self {
        let (Divisor::Static(d), Offset::Static(o)) = (self.divisor, self.offset) else {
            return self;
        };
        let common = self.terms.iter().try_fold(0, |g, t| match t.scale {
            Scale::Static(s) => Some(gcd(g, s)),
            Scale::Dynamic { .. } => None,
        });
        // `gcd(0, s)` seeds from the first coefficient; an all-`0` combination reports `0`.
        if !matches!(common, Some(g) if g % d == 0) {
            return self;
        }
        for term in self.terms.iter_mut() {
            term.scale = Scale::Static(term.scale.get() / d);
        }
        self.offset = Offset::Static(o.div_euclid(d as isize));
        self.divisor = Divisor::Static(1);
        self
    }

    /// What this physical axis is addressed by.
    pub(crate) fn addressed(&self) -> Addressed {
        match self.terms().first() {
            Some(term) => Addressed::By(term.axis),
            None => Addressed::Broadcast,
        }
    }

    pub(crate) fn terms(&self) -> &[AxisTerm] {
        &self.terms
    }

    /// How this axis's terms sit on it ([`Composition`]).
    pub(crate) fn composition(&self) -> Composition {
        self.composition
    }

    /// The `(axis, coefficient)` radices a disjoint map claims, coarsest first; empty otherwise.
    pub(crate) fn claimed_radices(&self) -> SmallVec<[(Axis, usize); Space::MAX_RANK]> {
        match self.composition {
            Composition::Overlapping => SmallVec::new(),
            Composition::Disjoint => self
                .terms
                .iter()
                .map(|t| (t.axis, t.scale.bound()))
                .collect(),
        }
    }

    /// The offset of this physical axis.
    pub fn offset(&self) -> Offset {
        self.offset
    }

    /// The divisor, [`Static(1)`](Divisor::Static) unless [`over`](Self::over) made it rational.
    pub(crate) fn divisor(&self) -> Divisor {
        self.divisor
    }

    /// Term `term`'s static step outside the rational floor, where `scale / divisor` is exact.
    pub(crate) fn static_offset_step(&self, term: usize) -> Option<usize> {
        match (self.divisor, self.terms[term].scale) {
            (Divisor::Static(d), Scale::Static(s)) if d > 1 && s.is_multiple_of(d) => Some(s / d),
            _ => None,
        }
    }

    /// Whether this axis divides by anything but `1`.
    pub(crate) fn is_rational(&self) -> bool {
        !self.divisor.is_unit()
    }

    /// The physical cell the logical origin lands on, `⌊offset / divisor⌋`; `None` if dynamic.
    pub(crate) fn origin(&self) -> Option<isize> {
        match (self.offset, self.divisor) {
            (Offset::Static(o), Divisor::Static(d)) => Some(o.div_euclid(d as isize)),
            _ => None,
        }
    }

    /// The division's starting phase, `offset - divisor * origin`; `None` if dynamic.
    pub(crate) fn residue(&self) -> Option<usize> {
        match (self.offset, self.divisor) {
            (Offset::Static(o), Divisor::Static(d)) => Some(o.rem_euclid(d as isize) as usize),
            _ => None,
        }
    }

    /// How many of this axis's coefficients are [`Dynamic`](Scale::Dynamic).
    pub(crate) fn dynamic_scale_count(&self) -> usize {
        self.terms.iter().filter(|t| t.scale.is_dynamic()).count()
    }

    /// Whether any of this axis's coefficients are [`Dynamic`](Scale::Dynamic).
    pub(crate) fn has_dynamic_scale(&self) -> bool {
        self.terms.iter().any(|t| t.scale.is_dynamic())
    }

    /// Whether `axis` addresses this physical axis at all.
    pub(crate) fn addresses(&self, axis: Axis) -> bool {
        self.terms.iter().any(|t| t.axis == axis)
    }

    /// `axis`'s coefficient, `0` when it does not address this axis. Panics if it is dynamic.
    pub fn scale(&self, axis: Axis) -> usize {
        self.terms
            .iter()
            .find(|t| t.axis == axis)
            .map_or(0, |t| t.scale.get())
    }

    /// The one [`Axis`] this map is the identity of, `None` otherwise.
    pub(crate) fn identity_axis(&self) -> Option<Axis> {
        match self.terms.as_slice() {
            [
                AxisTerm {
                    axis,
                    scale: Scale::Static(1),
                },
            ] if self.offset == Offset::Static(0) && self.divisor.is_unit() => Some(*axis),
            _ => None,
        }
    }

    /// The axes this buffer dim is made of, when it is one index over several axes laid end to
    /// end, like `K = KB·16 + KI` over `KI` and `KB`. Listed finest first: the finest at scale
    /// `1`, each next one at the product of the extents before it, no offset, no division. Such a
    /// dim stays in the buffer whenever each of its axes stays in its extent. A single axis at
    /// scale `1` is the simplest case ([`identity_axis`](Self::identity_axis)). `None` for any
    /// other map, or when an axis other than the last has no comptime extent in `space`.
    pub(crate) fn flattened_axes(
        &self,
        space: &Space,
    ) -> Option<SmallVec<[Axis; Space::MAX_RANK]>> {
        if self.terms.is_empty() || self.offset != Offset::Static(0) || !self.divisor.is_unit() {
            return None;
        }
        let mut terms = self
            .terms
            .iter()
            .map(|term| match term.scale {
                Scale::Static(scale) => Some((term.axis, scale)),
                Scale::Dynamic { .. } => None,
            })
            .collect::<Option<SmallVec<[(Axis, usize); Space::MAX_RANK]>>>()?;
        terms.sort_by_key(|&(_, scale)| scale);
        let mut next_scale = Some(1);
        for &(axis, scale) in &terms {
            if next_scale != Some(scale) || !space.contains(axis) {
                return None;
            }
            next_scale = match space.extent_raw(axis) {
                Extent::Static(extent) => Some(scale * extent),
                Extent::Dynamic => None,
            };
        }
        Some(terms.iter().map(|&(axis, _)| axis).collect())
    }

    /// Whether this physical axis is exactly `axis` at coefficient `1`, offset `0`, no division.
    pub(crate) fn is_identity(&self, axis: Axis) -> bool {
        self.identity_axis() == Some(axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);

    #[test]
    fn identity_is_exactly_one_axis_at_coefficient_one() {
        let id = PhysicalAxisMap::of(A);
        assert!(id.is_identity(A));
        assert!(!id.is_identity(B));
        assert_eq!(id.scale(A), 1);
        assert_eq!(id.scale(B), 0);
        assert_eq!(id.offset(), Offset::Static(0));

        let affine = PhysicalAxisMap::affine(&[(A, 2), (B, 3)]);
        assert!(!affine.is_identity(A));
        assert_eq!(affine.scale(A), 2);
        assert_eq!(affine.scale(B), 3);
        assert_eq!(affine.offset(), Offset::Static(0));
        assert!(!PhysicalAxisMap::affine(&[(A, 2)]).is_identity(A));
        assert!(PhysicalAxisMap::affine(&[(A, 1)]).is_identity(A));

        let with_offset = PhysicalAxisMap::affine(&[(A, 1)]).shifted(-2);
        assert!(!with_offset.is_identity(A));
        assert_eq!(with_offset.scale(A), 1);
        assert_eq!(with_offset.offset(), Offset::Static(-2));

        let with_dynamic_offset = PhysicalAxisMap::affine(&[(A, 1)]).shifted(Offset::Dynamic);
        assert!(!with_dynamic_offset.is_identity(A));
        assert_eq!(with_dynamic_offset.offset(), Offset::Dynamic);
        assert!(with_dynamic_offset.offset().is_dynamic());
        assert!(!with_dynamic_offset.has_dynamic_scale());
        assert_eq!(with_dynamic_offset.dynamic_scale_count(), 0);
    }

    /// The axes come back finest first, whatever order the terms are written in. A gap or
    /// overlap between them, an offset, a division, a dynamic scale, or a finer axis with no
    /// comptime extent gives `None`.
    #[test]
    fn flattened_axes_come_finest_first() {
        const C: Axis = Axis(2);
        let space = Space::new(&[(A, 4), (B, 16), (C, 3)]);
        let axes = |map: &PhysicalAxisMap| map.flattened_axes(&space).map(|axes| axes.to_vec());

        assert_eq!(axes(&PhysicalAxisMap::of(A)), Some(vec![A]));
        assert_eq!(
            axes(&PhysicalAxisMap::affine(&[(A, 16), (B, 1)])),
            Some(vec![B, A])
        );
        assert_eq!(
            axes(&PhysicalAxisMap::affine(&[(C, 64), (A, 16), (B, 1)])),
            Some(vec![B, A, C])
        );

        assert_eq!(axes(&PhysicalAxisMap::affine(&[(A, 32), (B, 1)])), None);
        assert_eq!(axes(&PhysicalAxisMap::affine(&[(A, 8), (B, 1)])), None);
        assert_eq!(axes(&PhysicalAxisMap::affine(&[(A, 16), (B, 2)])), None);
        assert_eq!(axes(&PhysicalAxisMap::of(A).shifted(1)), None);
        assert_eq!(axes(&PhysicalAxisMap::of(A).over(2)), None);
        assert_eq!(
            axes(&PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 4 })])),
            None
        );
        assert_eq!(axes(&PhysicalAxisMap::of(Axis(7))), None);

        let finer_dynamic = Space::new(&[(A, 4), (B, 16)]).with_dynamic(&[B]);
        let blocked = PhysicalAxisMap::affine(&[(A, 16), (B, 1)]);
        assert_eq!(blocked.flattened_axes(&finer_dynamic), None);
        let coarser_dynamic = Space::new(&[(A, 4), (B, 16)]).with_dynamic(&[A]);
        assert_eq!(
            blocked
                .flattened_axes(&coarser_dynamic)
                .map(|axes| axes.to_vec()),
            Some(vec![B, A])
        );
    }

    #[test]
    fn rational_axis_map_properties() {
        let map = PhysicalAxisMap::affine(&[(A, 100)]).shifted(-50).over(133);
        assert!(map.is_rational());
        assert_eq!(map.divisor(), Divisor::Static(133));
        assert_eq!(map.offset(), Offset::Static(-50));
        assert_eq!(map.origin(), Some((-50isize).div_euclid(133)));
        assert_eq!(map.residue(), Some((-50isize).rem_euclid(133) as usize));

        let dynamic_div = PhysicalAxisMap::affine(&[(A, 100)]).over(Divisor::Dynamic { min: 3 });
        assert!(dynamic_div.is_rational());
        assert!(dynamic_div.divisor().is_dynamic());
        assert_eq!(dynamic_div.origin(), None);
        assert_eq!(dynamic_div.residue(), None);
    }

    /// A rational map's terms whose coefficient the divisor divides step exactly.
    #[test]
    fn a_rational_map_factors_exact_static_terms_into_offsets() {
        let map = PhysicalAxisMap::affine(&[(A, 5), (B, 6)])
            .shifted(-2)
            .over(6);

        assert!(map.is_rational());
        assert_eq!(map.static_offset_step(0), None);
        assert_eq!(map.static_offset_step(1), Some(1));
        for output in 0..8isize {
            for tap in 0..3isize {
                assert_eq!(
                    (output * 5 + tap * 6 - 2).div_euclid(6),
                    (output * 5 - 2).div_euclid(6) + tap
                );
            }
        }

        let strided_tap = PhysicalAxisMap::affine(&[(A, 5), (B, 12)]).over(6);
        assert_eq!(strided_tap.static_offset_step(1), Some(2));

        let fractional_tap = PhysicalAxisMap::affine(&[(A, 5), (B, 2)]).over(3);
        assert_eq!(fractional_tap.static_offset_step(1), None);

        let dynamic_divisor =
            PhysicalAxisMap::affine(&[(A, 5), (B, 6)]).over(Divisor::Dynamic { min: 3 });
        assert_eq!(dynamic_divisor.static_offset_step(1), None);
    }

    #[test]
    #[should_panic(expected = "divisor of 0 does not map anywhere")]
    fn divisor_zero_panics() {
        PhysicalAxisMap::affine(&[(A, 1)]).over(0);
    }

    #[test]
    #[should_panic(expected = "state the whole divisor once")]
    fn a_second_over_is_refused() {
        PhysicalAxisMap::affine(&[(A, 4), (B, 6)]).over(4).over(3);
    }

    #[test]
    fn a_divisor_the_coefficients_cancel_reduces_away() {
        let map = PhysicalAxisMap::affine(&[(A, 8), (B, 4)]).over(4);
        assert!(!map.is_rational());
        assert_eq!(map.divisor(), Divisor::Static(1));
        assert_eq!(map.scale(A), 2);
        assert_eq!(map.scale(B), 1);
        assert_eq!(map.offset(), Offset::Static(0));

        assert!(PhysicalAxisMap::affine(&[(A, 4)]).over(4).is_identity(A));
    }

    /// `⌊(8a - 3)/4⌋` is `2a - 1`.
    #[test]
    fn reducing_floors_the_offset() {
        let map = PhysicalAxisMap::affine(&[(A, 8)]).shifted(-3).over(4);
        assert!(!map.is_rational());
        assert_eq!(map.scale(A), 2);
        assert_eq!(map.offset(), Offset::Static(-1));
        assert_eq!(map.origin(), Some(-1));
        assert_eq!(map.residue(), Some(0));
    }

    #[test]
    fn a_divisor_the_coefficients_do_not_cancel_stays() {
        assert!(
            PhysicalAxisMap::affine(&[(A, 4), (B, 6)])
                .over(6)
                .is_rational()
        );
        assert!(
            PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 2 }), (B, Scale::Static(4))])
                .over(4)
                .is_rational()
        );
        assert!(
            PhysicalAxisMap::scaled(&[(A, Scale::Static(4))])
                .shifted(Offset::Dynamic)
                .over(4)
                .is_rational()
        );
    }

    #[test]
    fn the_same_coefficients_partition_or_overlap() {
        assert_eq!(
            PhysicalAxisMap::disjoint(&[(A, 2), (B, 1)]).composition(),
            Composition::Disjoint
        );
        assert_eq!(
            PhysicalAxisMap::affine(&[(A, 2), (B, 1)]).composition(),
            Composition::Overlapping
        );
    }

    #[test]
    fn the_identity_partitions_whichever_door_minted_it() {
        assert_eq!(PhysicalAxisMap::of(A).composition(), Composition::Disjoint);
        assert_eq!(
            PhysicalAxisMap::affine(&[(A, 1)]).composition(),
            Composition::Disjoint
        );
        assert_eq!(
            PhysicalAxisMap::scaled(&[(A, Scale::Static(1))]).composition(),
            Composition::Disjoint
        );
        assert_eq!(
            PhysicalAxisMap::affine(&[(A, 2)]).composition(),
            Composition::Overlapping
        );
    }

    #[test]
    fn a_coarsening_is_never_a_partition() {
        assert_eq!(
            PhysicalAxisMap::of(A).over(4).composition(),
            Composition::Overlapping
        );
    }

    #[test]
    fn a_partition_claims_its_radices() {
        let map = PhysicalAxisMap::disjoint(&[(A, 32), (B, 1)]);
        assert_eq!(&map.claimed_radices()[..], &[(A, 32), (B, 1)]);
        assert!(
            PhysicalAxisMap::affine(&[(A, 32), (B, 1)])
                .claimed_radices()
                .is_empty()
        );
    }
}
