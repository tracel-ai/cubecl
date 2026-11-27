//! One physical axis of a buffer as an affine combination of dimensions, and its coefficients.

use cubecl::zspace::SmallVec;

use crate::space::extent::Extent;
use crate::{Dimension, Space, WalkUnits, algebra::gcd};

/// How far one unit of a dimension moves along one physical axis.
/// A `Static` coefficient is signed: a negative one steps the physical axis backward. A `Dynamic`
/// coefficient declares `max`, the largest value the launch may pass, and is never negative.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scale {
    Static(isize),
    Dynamic { max: usize },
}

impl Scale {
    /// The comptime coefficient; panics on `Dynamic`.
    pub fn get(self) -> isize {
        match self {
            Scale::Static(n) => n,
            Scale::Dynamic { .. } => {
                panic!(
                    "Scale::get: this coefficient is Dynamic; its value is only known at runtime"
                )
            }
        }
    }

    /// The largest absolute value this coefficient can take: itself, or the declared `max`.
    pub fn bound(self) -> usize {
        match self {
            Scale::Static(n) => n.unsigned_abs(),
            Scale::Dynamic { max } => max,
        }
    }

    pub(crate) fn is_negative(self) -> bool {
        matches!(self, Scale::Static(n) if n < 0)
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

/// One digit of a coordinate: `(coordinate / place_value) % extent`, or the whole quotient for
/// the coarsest digit ([`Quotient::Whole`]), so a position past the coordinate's extent never wraps
/// onto another position's data; the operand bounding it masks it. A coordinate folded from
/// several dimensions ([`Projection::folded`](crate::Projection::folded)) is read as its digits,
/// and so is a coordinate a buffer stores as several pieces
/// ([`Projection::pieces_as_digits`](crate::Projection::pieces_as_digits)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Digit {
    /// The product of the extents of the digits inside this one.
    pub place_value: Radix,
    pub extent: Radix,
    pub quotient: Quotient,
}

/// A digit's place value or extent: known where the kernel is written, or read off the buffer
/// when it runs, from the runtime map's digit radices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Radix {
    Static(usize),
    Dynamic,
}

/// A radix read in the kernel's expansion, where it is only ever known at compile time.
mod radix_expand {
    use super::Radix;
    use crate::comptime_only;
    use cubecl::prelude::*;

    comptime_only!(Radix);
}

/// How one term of a physical axis moves when its dimension is walked one count of a compile-time
/// unit ([`PhysicalAxisMap::term_step`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum TermStep {
    /// By this constant, in the physical axis's units.
    Adds(isize),
    /// By the term's runtime coefficient times this many coordinates.
    AddsCoefficient(usize),
    /// Not at all.
    Holds,
    /// By no constant: the term is evaluated again where the walk lands.
    Reevaluates,
}

impl Digit {
    /// How this digit moves when its coordinate moves `coords_per_count` a count, in the digit's
    /// own units, before its term's coefficient: a coarsest digit adds the quotient once its place
    /// value divides the unit, a digit inside another holds once its period does, and any other
    /// digit, or one whose radix the rule reads is [`Radix::Dynamic`], is evaluated again. A
    /// coarsest digit's extent is never read.
    pub(crate) fn step(&self, coords_per_count: usize) -> TermStep {
        match (self.quotient, self.place_value, self.extent) {
            (Quotient::Whole, Radix::Static(place), _)
                if coords_per_count.is_multiple_of(place) =>
            {
                TermStep::Adds((coords_per_count / place) as isize)
            }
            (Quotient::Modulo, Radix::Static(place), Radix::Static(extent))
                if coords_per_count.is_multiple_of(place * extent) =>
            {
                TermStep::Holds
            }
            _ => TermStep::Reevaluates,
        }
    }
}

/// What a [`Digit`] keeps of `coordinate / place_value`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Quotient {
    /// Its remainder by the digit's extent: a dimension folded inside another.
    Modulo,
    /// All of it: the outermost dimension of the fold.
    Whole,
}

/// One dimension's contribution to one physical axis: its coordinate, or one [`Digit`] of it,
/// times `scale`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct DimensionTerm {
    pub dimension: Dimension,
    pub scale: Scale,
    pub digit: Option<Digit>,
}

impl DimensionTerm {
    /// The largest coordinate the term takes over a region `extent` long. For a digit this is
    /// its own `extent - 1`, an upper bound whatever extent and start the region has, since a
    /// window is never moved along a digit.
    pub(crate) fn largest_coordinate(&self, extent: usize) -> usize {
        match self.digit.map(|digit| digit.extent) {
            Some(Radix::Static(digits)) => digits - 1,
            Some(Radix::Dynamic) => panic!(
                "DimensionTerm::largest_coordinate: {:?}'s digit has an extent read off the buffer, so \
                 its span is not known where the kernel is written",
                self.dimension
            ),
            None => extent - 1,
        }
    }
}

/// Whether a physical axis's terms can land on the same cell, as stated by the caller.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Composition {
    /// No two logical positions share a cell: the terms partition the dimension.
    Disjoint,
    /// No two logical positions share a cell, but the cells start at an offset: a partition
    /// moved along its dimension, as a window of rows read from a dynamic position is. A write
    /// through it aliases nothing; it is read as an N-D view, never as a partition that starts
    /// at 0.
    Translated,
    /// Two positions may land on the same cell, as with a stencil or a resample.
    Overlapping,
}

/// One buffer axis as `(Σ coordinate(dimension) * scale + offset) / divisor`, floored.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxisMap {
    terms: SmallVec<[DimensionTerm; Space::MAX_RANK]>,
    offset: Offset,
    divisor: Divisor,
    composition: Composition,
    /// Which piece of its dimension an axis stored as one of several is, the coarsest the first.
    piece: usize,
}

/// What a physical axis is addressed by.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Addressed {
    /// The leading dimension of this map's terms, naming its coordinate group.
    By(Dimension),
    /// A digit of a folded dimension: a coordinate of its own, never grouped with another axis.
    ByDigit,
    /// Nothing: every position resolves to the same element.
    Broadcast,
}

impl PhysicalAxisMap {
    /// A map addressing no dimension: every position resolves to the same element.
    pub fn broadcast() -> Self {
        PhysicalAxisMap::scaled(&[])
    }

    /// `dimension` whole, or the coarsest of the pieces it is stored as.
    pub fn of(dimension: Dimension) -> Self {
        PhysicalAxisMap::piece_of(dimension, 0)
    }

    /// One of the pieces `dimension` is stored as, an axis each, `piece` of them coarser.
    pub fn piece_of(dimension: Dimension, piece: usize) -> Self {
        PhysicalAxisMap {
            terms: SmallVec::from_slice(&[DimensionTerm {
                dimension,
                scale: Scale::Static(1),
                digit: None,
            }]),
            offset: Offset::Static(0),
            divisor: Divisor::Static(1),
            composition: Composition::Disjoint,
            piece,
        }
    }

    /// One `digit` of `dimension`'s coordinate at coefficient `1`: a buffer axis storing a piece
    /// of the coordinate, read as the digit it is
    /// ([`Projection::pieces_as_digits`](crate::Projection::pieces_as_digits)).
    pub(crate) fn digit_of(dimension: Dimension, digit: Digit) -> Self {
        PhysicalAxisMap {
            terms: SmallVec::from_slice(&[DimensionTerm {
                dimension,
                scale: Scale::Static(1),
                digit: Some(digit),
            }]),
            offset: Offset::Static(0),
            divisor: Divisor::Static(1),
            composition: Composition::Overlapping,
            piece: 0,
        }
    }

    /// This physical axis partitioned by several logical ones, coarsest first:
    /// `disjoint(&[(KB, 32), (KI, 1)])` is `k = kb * 32 + ki`.
    pub fn disjoint(terms: &[(Dimension, usize)]) -> Self {
        let mut map = Self::affine(terms);
        map.composition = Composition::Disjoint;
        map
    }

    /// An affine combination, e.g. `affine(&[(Oh, stride), (Rh, dilation)])`.
    pub fn affine(terms: &[(Dimension, usize)]) -> Self {
        let terms: SmallVec<[(Dimension, Scale); Space::MAX_RANK]> = terms
            .iter()
            .map(|&(dimension, scale)| {
                (
                    dimension,
                    Scale::Static(isize::try_from(scale).expect("a coefficient fits an isize")),
                )
            })
            .collect();
        Self::scaled(&terms)
    }

    /// [`affine`](Self::affine) over explicit [`Scale`]s, for runtime coefficients.
    pub fn scaled(terms: &[(Dimension, Scale)]) -> Self {
        for &(dimension, scale) in terms {
            assert!(
                scale.bound() > 0,
                "PhysicalAxisMap: {dimension:?}'s coefficient is bounded at 0, so it addresses no cell; \
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
                .map(|&(dimension, scale)| DimensionTerm {
                    dimension,
                    scale,
                    digit: None,
                })
                .collect(),
            offset: Offset::Static(0),
            divisor: Divisor::Static(1),
            composition,
            piece: 0,
        }
    }

    /// The same combination plus a signed constant or dynamic `offset`, added before any
    /// [`over`](Self::over). A partition shifted stays one cell a position, moved:
    /// `Composition::Translated`.
    pub fn shifted(mut self, offset: impl Into<Offset>) -> Self {
        let offset = offset.into();
        assert!(
            self.divisor.is_unit() && self.offset == Offset::Static(0),
            "PhysicalAxisMap::shifted: shift once, before `over`"
        );
        if offset != Offset::Static(0) {
            self.composition = match self.composition {
                Composition::Disjoint | Composition::Translated => Composition::Translated,
                Composition::Overlapping => Composition::Overlapping,
            };
        }
        self.offset = offset;
        self
    }

    /// The same map with each term on a dimension of `fold` read as its digit of `into`
    /// ([`Projection::folded`](crate::Projection::folded)), `fold` outermost first.
    pub(crate) fn folded(mut self, into: Dimension, fold: &[(Dimension, usize)]) -> Self {
        let mut read = false;
        for term in self.terms.iter_mut() {
            let Some(at) = fold.iter().position(|&(d, _)| d == term.dimension) else {
                continue;
            };
            assert!(
                !read,
                "PhysicalAxisMap::folded: this axis reads two dimensions of the fold; one digit an \
                 axis"
            );
            assert!(
                matches!(term.scale, Scale::Static(s) if s > 0),
                "PhysicalAxisMap::folded: {:?} steps by {:?}; a digit takes a static forward \
                 coefficient",
                term.dimension,
                term.scale
            );
            read = true;
            *term = DimensionTerm {
                dimension: into,
                scale: term.scale,
                digit: Some(Digit {
                    place_value: Radix::Static(fold[at + 1..].iter().map(|&(_, e)| e).product()),
                    extent: Radix::Static(fold[at].1),
                    quotient: match at {
                        0 => Quotient::Whole,
                        _ => Quotient::Modulo,
                    },
                }),
            };
        }
        if read {
            assert!(
                self.divisor.is_unit(),
                "PhysicalAxisMap::folded: a digit under a division is refused"
            );
            self.composition = Composition::Overlapping;
        }
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
        assert!(
            divisor.is_unit() || !self.has_negative_scale(),
            "PhysicalAxisMap::over: a negative coefficient under a division floors the wrong way \
             and is refused"
        );
        assert!(
            divisor.is_unit() || self.digit_dimension().is_none(),
            "PhysicalAxisMap::over: a digit under a division is refused"
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
            Scale::Static(s) => Some(gcd(g, s.unsigned_abs())),
            Scale::Dynamic { .. } => None,
        });
        // `gcd(0, s)` seeds from the first coefficient; an all-`0` combination reports `0`.
        if !matches!(common, Some(g) if g % d == 0) {
            return self;
        }
        for term in self.terms.iter_mut() {
            term.scale = Scale::Static(term.scale.get() / d as isize);
        }
        self.offset = Offset::Static(o.div_euclid(d as isize));
        self.divisor = Divisor::Static(1);
        self
    }

    /// What this physical axis is addressed by.
    pub(crate) fn addressed(&self) -> Addressed {
        if self.digit_dimension().is_some() {
            return Addressed::ByDigit;
        }
        match self.terms().first() {
            Some(term) => Addressed::By(term.dimension),
            None => Addressed::Broadcast,
        }
    }

    pub(crate) fn terms(&self) -> &[DimensionTerm] {
        &self.terms
    }

    /// The dimension this axis reads a digit of, `None` when it reads none.
    pub(crate) fn digit_dimension(&self) -> Option<Dimension> {
        self.terms
            .iter()
            .find(|term| term.digit.is_some())
            .map(|term| term.dimension)
    }

    /// How many of the pieces its dimension is stored as are coarser.
    pub(crate) fn piece(&self) -> usize {
        self.piece
    }

    /// This axis and the other pieces of its dimension as one axis.
    pub(crate) fn whole(mut self) -> Self {
        self.piece = 0;
        self
    }

    /// How this dimension's terms sit on it ([`Composition`]).
    pub(crate) fn composition(&self) -> Composition {
        self.composition
    }

    /// The `(dimension, coefficient)` radices a disjoint map claims, coarsest first; empty
    /// otherwise.
    pub(crate) fn claimed_radices(&self) -> SmallVec<[(Dimension, usize); Space::MAX_RANK]> {
        match self.composition {
            Composition::Overlapping | Composition::Translated => SmallVec::new(),
            Composition::Disjoint => self
                .terms
                .iter()
                .map(|t| (t.dimension, t.scale.bound()))
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

    /// How term `term` moves when its dimension moves `coords_per_count` a count, in this axis's
    /// units ([`TermStep`]): a digit as [`Digit::step`] says, scaled by its coefficient; a plain
    /// term on an integer axis by its coefficient; a plain term under a static divisor by the
    /// quotient once the divisor divides its move, every other one evaluated again.
    ///
    /// # Assumptions
    ///
    /// - `term` is one of this axis's terms.
    /// - `coords_per_count` counts the units the term's coordinate counts.
    ///
    /// # Guarantees
    ///
    /// - For every coordinate `x` and count `n`, the term's value at `x + n·coords_per_count`
    ///   less its value at `x` is `n·c` under `Adds(c)`, `n·u` times its runtime coefficient
    ///   under `AddsCoefficient(u)`, and `0` under `Holds`; under a division, the axis's value
    ///   moves so whatever its other terms.
    /// - It answers [`TermStep::Reevaluates`] only where no such constant exists or a radix or
    ///   divisor it needs is only known at runtime.
    pub(crate) fn term_step(&self, term: usize, coords_per_count: usize) -> TermStep {
        let at = self.terms[term];
        match at.digit.map(|digit| digit.step(coords_per_count)) {
            Some(TermStep::Adds(digits)) => TermStep::Adds(digits * at.scale.get()),
            Some(step) => step,
            None => self.plain_step(at.scale, coords_per_count),
        }
    }

    /// Whether walking `walk` moves a term of this axis that stays inside its floor, so the
    /// floor is taken again where the walk lands.
    pub(crate) fn reevaluates_floor(&self, walk: &WalkUnits) -> bool {
        self.is_rational()
            && self.terms.iter().enumerate().any(|(t, term)| {
                walk.coords_per_count(term.dimension)
                    .is_some_and(|unit| self.term_step(t, unit) == TermStep::Reevaluates)
            })
    }

    /// [`term_step`](Self::term_step) of a plain term of coefficient `scale`.
    fn plain_step(&self, scale: Scale, coords_per_count: usize) -> TermStep {
        let moved = coords_per_count as isize;
        match (self.divisor, scale) {
            (Divisor::Static(1), Scale::Static(s)) => TermStep::Adds(moved * s),
            (Divisor::Static(1), Scale::Dynamic { .. }) => {
                TermStep::AddsCoefficient(coords_per_count)
            }
            (Divisor::Static(d), Scale::Static(s)) if (moved * s).rem_euclid(d as isize) == 0 => {
                TermStep::Adds(moved * s / d as isize)
            }
            _ => TermStep::Reevaluates,
        }
    }

    /// Term `term`'s static step outside the rational floor, where `scale / divisor` is exact.
    pub(crate) fn static_offset_step(&self, term: usize) -> Option<usize> {
        match (self.divisor, self.terms[term].scale) {
            (Divisor::Static(d), Scale::Static(s)) if d > 1 => usize::try_from(s)
                .ok()
                .filter(|s| s.is_multiple_of(d))
                .map(|s| s / d),
            _ => None,
        }
    }

    /// Whether this dimension divides by anything but `1`.
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

    /// How many of this dimension's coefficients are [`Dynamic`](Scale::Dynamic).
    pub(crate) fn dynamic_scale_count(&self) -> usize {
        self.terms.iter().filter(|t| t.scale.is_dynamic()).count()
    }

    /// Whether any of this dimension's coefficients steps it backward.
    pub(crate) fn has_negative_scale(&self) -> bool {
        self.terms.iter().any(|t| t.scale.is_negative())
    }

    /// Whether any of this dimension's coefficients are [`Dynamic`](Scale::Dynamic).
    pub(crate) fn has_dynamic_scale(&self) -> bool {
        self.terms.iter().any(|t| t.scale.is_dynamic())
    }

    /// Whether `dimension` addresses this physical axis at all.
    pub(crate) fn addresses(&self, dimension: Dimension) -> bool {
        self.terms.iter().any(|t| t.dimension == dimension)
    }

    /// `dimension`'s coefficient, `0` when it does not address this dimension. Panics if it is
    /// dynamic.
    pub fn scale(&self, dimension: Dimension) -> isize {
        self.terms
            .iter()
            .find(|t| t.dimension == dimension)
            .map_or(0, |t| t.scale.get())
    }

    /// The one [`Dimension`] this map is the identity of, `None` otherwise.
    pub(crate) fn identity_dimension(&self) -> Option<Dimension> {
        match self.terms.as_slice() {
            [
                DimensionTerm {
                    dimension,
                    scale: Scale::Static(1),
                    digit: None,
                },
            ] if self.offset == Offset::Static(0) && self.divisor.is_unit() => Some(*dimension),
            _ => None,
        }
    }

    /// The dimensions this buffer axis is made of, when it is one index over several dimensions
    /// laid end to end, like `K = KB·16 + KI` over `KI` and `KB`. Listed finest first: the finest
    /// at scale `1`, each next one at the product of the extents before it, no offset, no division.
    /// Such a axis stays in the buffer whenever each of its dimensions stays in its extent. A
    /// single dimension at scale `1` is the simplest case
    /// ([`identity_dimension`](Self::identity_dimension)). `None` for any other map, or when a
    /// dimension other than the last has no comptime extent in `space`.
    pub(crate) fn flattened_dimensions(
        &self,
        space: &Space,
    ) -> Option<SmallVec<[Dimension; Space::MAX_RANK]>> {
        if self.terms.is_empty()
            || self.offset != Offset::Static(0)
            || !self.divisor.is_unit()
            || self.digit_dimension().is_some()
        {
            return None;
        }
        let mut terms = self
            .terms
            .iter()
            .map(|term| match term.scale {
                Scale::Static(scale) => usize::try_from(scale).ok().map(|s| (term.dimension, s)),
                Scale::Dynamic { .. } => None,
            })
            .collect::<Option<SmallVec<[(Dimension, usize); Space::MAX_RANK]>>>()?;
        terms.sort_by_key(|&(_, scale)| scale);
        let mut next_scale = Some(1);
        for &(dimension, scale) in &terms {
            if next_scale != Some(scale) || !space.contains(dimension) {
                return None;
            }
            next_scale = match space.extent_raw(dimension) {
                Extent::Static(extent) => Some(scale * extent),
                Extent::Dynamic => None,
            };
        }
        Some(terms.iter().map(|&(dimension, _)| dimension).collect())
    }

    /// Whether this physical axis is exactly `dimension` at coefficient `1`, offset `0`, no
    /// division.
    pub(crate) fn is_identity(&self, dimension: Dimension) -> bool {
        self.identity_dimension() == Some(dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Dimension = Dimension(0);
    const B: Dimension = Dimension(1);

    #[test]
    fn a_plain_affine_term_steps_by_its_coefficient() {
        let map = PhysicalAxisMap::affine(&[(A, 2), (B, 3)]);
        assert_eq!(map.term_step(0, 1), TermStep::Adds(2));
        assert_eq!(map.term_step(1, 4), TermStep::Adds(12));
    }

    /// A digit of place `7`: whole, it adds once the unit is a multiple of `7`.
    #[test]
    fn a_whole_digit_adds_when_the_unit_is_a_multiple_of_its_place_value() {
        let digit = Digit {
            place_value: Radix::Static(7),
            extent: Radix::Static(5),
            quotient: Quotient::Whole,
        };
        assert_eq!(digit.step(14), TermStep::Adds(2));
    }

    /// A digit of place `2` inside another, `3` of it: one count of `6` or `12` wraps it whole.
    #[test]
    fn a_modulo_digit_holds_when_the_unit_spans_its_period() {
        let digit = Digit {
            place_value: Radix::Static(2),
            extent: Radix::Static(3),
            quotient: Quotient::Modulo,
        };
        assert_eq!(digit.step(6), TermStep::Holds);
        assert_eq!(digit.step(12), TermStep::Holds);
    }

    #[test]
    fn a_digit_reevaluates_when_its_place_does_not_divide_the_unit() {
        let whole = Digit {
            place_value: Radix::Static(7),
            extent: Radix::Static(5),
            quotient: Quotient::Whole,
        };
        assert_eq!(whole.step(3), TermStep::Reevaluates);
        // A multiple of its place, not of its period.
        let inside = Digit {
            place_value: Radix::Static(2),
            extent: Radix::Static(7),
            quotient: Quotient::Modulo,
        };
        assert_eq!(inside.step(4), TermStep::Reevaluates);
    }

    /// The coarsest digit never reads its extent, so the buffer's runtime count of its tiles does
    /// not stop it from adding; a digit inside another reads both radices.
    #[test]
    fn a_dynamic_radix_reevaluates_unless_the_rule_never_reads_it() {
        let inside = Digit {
            place_value: Radix::Static(1),
            extent: Radix::Dynamic,
            quotient: Quotient::Modulo,
        };
        assert_eq!(inside.step(16), TermStep::Reevaluates);
        let coarsest = Digit {
            place_value: Radix::Static(16),
            extent: Radix::Dynamic,
            quotient: Quotient::Whole,
        };
        assert_eq!(coarsest.step(32), TermStep::Adds(2));
        let unstated = Digit {
            place_value: Radix::Dynamic,
            extent: Radix::Dynamic,
            quotient: Quotient::Whole,
        };
        assert_eq!(unstated.step(16), TermStep::Reevaluates);
    }

    /// `⌊(3a + 4b)/2⌋`: `a` walked by two moves the floor by three, by one it does not move it by
    /// a constant; `b` moves it by two a count whatever the unit.
    #[test]
    fn a_floored_term_adds_once_the_unit_makes_it_whole() {
        let map = PhysicalAxisMap::affine(&[(A, 3), (B, 4)]).over(2);
        assert_eq!(map.term_step(0, 2), TermStep::Adds(3));
        assert_eq!(map.term_step(0, 1), TermStep::Reevaluates);
        assert_eq!(map.term_step(1, 1), TermStep::Adds(2));
        assert!(map.reevaluates_floor(&WalkUnits::along(A, 1)));
        assert!(!map.reevaluates_floor(&WalkUnits::over(&[(A, 2), (B, 1)])));
    }

    #[test]
    fn a_dynamic_divisor_reevaluates_and_a_dynamic_coefficient_adds_its_coefficient() {
        let map = PhysicalAxisMap::scaled(&[(A, Scale::Static(2)), (B, Scale::Dynamic { max: 4 })]);
        assert_eq!(map.term_step(1, 3), TermStep::AddsCoefficient(3));
        let divided = PhysicalAxisMap::affine(&[(A, 4)]).over(Divisor::Dynamic { min: 2 });
        assert_eq!(divided.term_step(0, 2), TermStep::Reevaluates);
    }

    /// Every step [`PhysicalAxisMap::term_step`] answers is the difference of the axis's values at
    /// both ends of the walk, wherever it starts: digits of every place, extent and coefficient,
    /// and plain terms under every divisor.
    #[test]
    fn term_step_matches_evaluating_the_axis() {
        let digit_value = |digit: Digit, x: usize| {
            let (Radix::Static(place), Radix::Static(extent)) = (digit.place_value, digit.extent)
            else {
                unreachable!("only static radices are evaluated on the host")
            };
            match digit.quotient {
                Quotient::Whole => x / place,
                Quotient::Modulo => x / place % extent,
            }
        };
        for place in 1..5 {
            for extent in 1..5 {
                for quotient in [Quotient::Whole, Quotient::Modulo] {
                    for scale in 1..4isize {
                        let digit = Digit {
                            place_value: Radix::Static(place),
                            extent: Radix::Static(extent),
                            quotient,
                        };
                        let map = PhysicalAxisMap {
                            terms: SmallVec::from_slice(&[DimensionTerm {
                                dimension: A,
                                scale: Scale::Static(scale),
                                digit: Some(digit),
                            }]),
                            offset: Offset::Static(0),
                            divisor: Divisor::Static(1),
                            composition: Composition::Overlapping,
                            piece: 0,
                        };
                        for unit in 1..13 {
                            let step = map.term_step(0, unit);
                            for x in 0..24 {
                                for n in 0..4 {
                                    let moved = (digit_value(digit, x + n * unit) as isize
                                        - digit_value(digit, x) as isize)
                                        * scale;
                                    match step {
                                        TermStep::Adds(c) => assert_eq!(moved, n as isize * c),
                                        TermStep::Holds => assert_eq!(moved, 0),
                                        TermStep::AddsCoefficient(_) => {
                                            unreachable!("a digit's coefficient is static")
                                        }
                                        TermStep::Reevaluates => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for divisor in 1..5usize {
            for scale in 0..6isize {
                let plain = |dimension, scale| DimensionTerm {
                    dimension,
                    scale: Scale::Static(scale),
                    digit: None,
                };
                // Built whole, as `over` would reduce a divisor every coefficient cancels.
                let map = PhysicalAxisMap {
                    terms: SmallVec::from_slice(&[plain(A, scale), plain(B, 3)]),
                    offset: Offset::Static(0),
                    divisor: Divisor::Static(divisor),
                    composition: Composition::Overlapping,
                    piece: 0,
                };
                let axis = |a: isize, b: isize| (a * scale + b * 3).div_euclid(divisor as isize);
                for unit in 1..7 {
                    let step = map.term_step(0, unit);
                    for (a, b) in (0..6).flat_map(|a| (0..6).map(move |b| (a, b))) {
                        for n in 0..4isize {
                            let moved = axis(a + n * unit as isize, b) - axis(a, b);
                            match step {
                                TermStep::Adds(c) => assert_eq!(moved, n * c),
                                TermStep::Holds => assert_eq!(moved, 0),
                                TermStep::AddsCoefficient(_) => {
                                    unreachable!("a static coefficient adds a constant")
                                }
                                TermStep::Reevaluates => {}
                            }
                        }
                    }
                    // A divisor of the move is a constant step.
                    let whole = (unit as isize * scale).rem_euclid(divisor as isize) == 0;
                    assert_eq!(step != TermStep::Reevaluates, whole);
                }
            }
        }
    }

    #[test]
    fn identity_is_exactly_one_dimension_at_coefficient_one() {
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

    /// `data = out + pad - tap * dilation`: the tap steps backward.
    #[test]
    fn a_backward_coefficient_stays_signed_and_never_flattens() {
        let map =
            PhysicalAxisMap::scaled(&[(A, Scale::Static(1)), (B, Scale::Static(-2))]).shifted(3);
        assert_eq!(map.scale(B), -2);
        assert_eq!(map.term_step(1, 1), TermStep::Adds(-2));
        assert!(map.has_negative_scale());
        assert!(!PhysicalAxisMap::scaled(&[(A, Scale::Static(-1))]).is_identity(A));

        let space = Space::new(&[(A, 4), (B, 16)]);
        let flat = PhysicalAxisMap::scaled(&[(A, Scale::Static(-16)), (B, Scale::Static(1))]);
        assert_eq!(flat.flattened_dimensions(&space), None);
    }

    #[test]
    #[should_panic(expected = "negative coefficient under a division")]
    fn a_backward_coefficient_under_a_division_is_refused() {
        PhysicalAxisMap::scaled(&[(A, Scale::Static(-4))]).over(2);
    }

    /// The dimensions come back finest first, whatever order the terms are written in. A gap or
    /// overlap between them, an offset, a division, a dynamic scale, or a finer dimension with no
    /// comptime extent gives `None`.
    #[test]
    fn flattened_dimensions_come_finest_first() {
        const C: Dimension = Dimension(2);
        let space = Space::new(&[(A, 4), (B, 16), (C, 3)]);
        let dimensions = |map: &PhysicalAxisMap| {
            map.flattened_dimensions(&space)
                .map(|dimensions| dimensions.to_vec())
        };

        assert_eq!(dimensions(&PhysicalAxisMap::of(A)), Some(vec![A]));
        assert_eq!(
            dimensions(&PhysicalAxisMap::affine(&[(A, 16), (B, 1)])),
            Some(vec![B, A])
        );
        assert_eq!(
            dimensions(&PhysicalAxisMap::affine(&[(C, 64), (A, 16), (B, 1)])),
            Some(vec![B, A, C])
        );

        assert_eq!(
            dimensions(&PhysicalAxisMap::affine(&[(A, 32), (B, 1)])),
            None
        );
        assert_eq!(
            dimensions(&PhysicalAxisMap::affine(&[(A, 8), (B, 1)])),
            None
        );
        assert_eq!(
            dimensions(&PhysicalAxisMap::affine(&[(A, 16), (B, 2)])),
            None
        );
        assert_eq!(dimensions(&PhysicalAxisMap::of(A).shifted(1)), None);
        assert_eq!(dimensions(&PhysicalAxisMap::of(A).over(2)), None);
        assert_eq!(
            dimensions(&PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 4 })])),
            None
        );
        assert_eq!(dimensions(&PhysicalAxisMap::of(Dimension(7))), None);

        let finer_dynamic = Space::new(&[(A, 4), (B, 16)]).with_dynamic(&[B]);
        let blocked = PhysicalAxisMap::affine(&[(A, 16), (B, 1)]);
        assert_eq!(blocked.flattened_dimensions(&finer_dynamic), None);
        let coarser_dynamic = Space::new(&[(A, 4), (B, 16)]).with_dynamic(&[A]);
        assert_eq!(
            blocked
                .flattened_dimensions(&coarser_dynamic)
                .map(|dimensions| dimensions.to_vec()),
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

    /// A partition shifted is one cell a position still, moved: a write through it aliases
    /// nothing, but it claims no radices, which describe a partition from 0. A shifted overlap
    /// stays one.
    #[test]
    fn a_shifted_partition_is_translated_and_a_shifted_overlap_is_not() {
        for offset in [Offset::Static(3), Offset::Dynamic] {
            let moved = PhysicalAxisMap::disjoint(&[(A, 2), (B, 1)]).shifted(offset);
            assert_eq!(moved.composition(), Composition::Translated);
            assert!(moved.claimed_radices().is_empty());
            assert_eq!(
                PhysicalAxisMap::of(A).shifted(offset).composition(),
                Composition::Translated
            );
            assert_eq!(
                PhysicalAxisMap::affine(&[(A, 2), (B, 1)])
                    .shifted(offset)
                    .composition(),
                Composition::Overlapping
            );
        }
        assert_eq!(
            PhysicalAxisMap::of(A).shifted(0).composition(),
            Composition::Disjoint
        );
    }

    #[test]
    fn a_coarsening_is_never_a_partition() {
        assert_eq!(
            PhysicalAxisMap::of(A).over(4).composition(),
            Composition::Overlapping
        );
    }

    const C: Dimension = Dimension(2);
    const P: Dimension = Dimension(3);
    /// `P = a * 7 + b`, `a` under 5.
    const FOLD: [(Dimension, usize); 2] = [(A, 5), (B, 7)];

    /// `h = a * 2 + tap - 1` folded: `a`'s term becomes the outermost digit of `P`, in place, and
    /// the axis is a coordinate of its own.
    #[test]
    fn a_folded_term_reads_its_digit_in_place() {
        let map = PhysicalAxisMap::affine(&[(A, 2), (C, 1)])
            .shifted(-1)
            .folded(P, &FOLD);
        assert_eq!(map.digit_dimension(), Some(P));
        assert_eq!(map.addressed(), Addressed::ByDigit);
        assert_eq!(
            map.terms()[0].digit,
            Some(Digit {
                place_value: Radix::Static(7),
                extent: Radix::Static(5),
                quotient: Quotient::Whole
            })
        );
        assert_eq!(map.scale(P), 2);
        assert_eq!(map.offset(), Offset::Static(-1));
        assert_eq!(map.term_step(0, 1), TermStep::Reevaluates);
        assert_eq!(map.term_step(0, 7), TermStep::Adds(2));
        assert_eq!(map.term_step(1, 1), TermStep::Adds(1));
        assert_eq!(map.composition(), Composition::Overlapping);
        assert!(map.claimed_radices().is_empty());
        assert_eq!(map.terms()[0].largest_coordinate(64), 4);
        assert_eq!(map.terms()[1].largest_coordinate(3), 2);

        let columns = PhysicalAxisMap::affine(&[(C, 1), (B, 1)]).folded(P, &FOLD);
        assert_eq!(
            columns.terms()[1].digit,
            Some(Digit {
                place_value: Radix::Static(1),
                extent: Radix::Static(7),
                quotient: Quotient::Modulo
            })
        );
        assert_eq!(columns.addressed(), Addressed::ByDigit);
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
