//! In-kernel evaluation of a [`Projection`] ([`ProjectionInKernel`]) and the walks over it
//! ([`Anchor`]).

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// A [`Layout`] mapping a tile's logical coordinate to its window's physical one:
/// `phys[pa] = (Σ logical[dimension] * scale + residue) / divisor`, constant offsets excluded; a
/// digit term reads its digit of `logical[dimension]` plus the tile's start along it.
///
/// A walk ([`WalkUnits`]) reads it once where the walk starts ([`anchor`](Self::anchor)), then at
/// every position the walk lands on from that start ([`advance`](Self::advance)): each term the
/// walk moves by a constant ([`TermStep`]) costs one product of a count by a compile-time constant
/// and one add, so a count known where the kernel is written leaves the start plus a constant.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct ProjectionInKernel {
    /// Per-logical-dimension extents in space order, the innermost in lines.
    shape: Coords<u32>,
    /// The runtime coefficients, division residues, digit starts and digit radices.
    map: RuntimeMap,
    #[cube(comptime)]
    space: Space,
    #[cube(comptime)]
    projection: Projection,
    /// The line width the innermost physical axis is addressed in.
    #[cube(comptime)]
    width: usize,
}

/// Where a walk over a [`ProjectionInKernel`] starts, evaluated once
/// ([`ProjectionInKernel::anchor`]): what [`advance`](ProjectionInKernel::advance) adds each
/// position's constants to, and what it reads again the terms that move by none.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Anchor {
    /// Each physical axis's value at the start without the terms the walk evaluates again, then
    /// the numerator of each floor it takes again, then the start's coordinate along each
    /// dimension a digit is evaluated again along.
    words: Coords<u32>,
}

#[cube]
impl Anchor {
    /// Store this anchor's words into `words` from `at` on, to read it back later with
    /// [`load_from`](Anchor::load_from).
    pub(crate) fn store_into(&self, words: &mut Array<u32>, at: usize) {
        #[unroll]
        for w in 0..self.words.len() {
            words[at + w] = self.words.at(w);
        }
    }

    /// The anchor `len` words long stored in `words` from `at` on
    /// ([`ProjectionInKernel::anchor_len`]).
    pub(crate) fn load_from(words: &Array<u32>, at: usize, #[comptime] len: usize) -> Anchor {
        let mut stored = Coords::<u32>::new();
        #[unroll]
        for w in 0..len {
            stored.push(words[at + w]);
        }
        Anchor { words: stored }
    }
}

#[cube]
impl ProjectionInKernel {
    pub(crate) fn new(
        shape: Coords<u32>,
        map: RuntimeMap,
        #[comptime] space: Space,
        #[comptime] projection: Projection,
        #[comptime] width: usize,
    ) -> Self {
        let rank = shape.len();
        comptime!(assert!(
            rank == space.rank(),
            "ProjectionInKernel: shape has {rank} entries but the space spans {} dimensions",
            space.rank()
        ));
        let given = map.coefficients.len();
        comptime!(assert!(
            given == projection.dynamic_coefficient_count(),
            "ProjectionInKernel: the projection has {} Dynamic coefficients and divisors but {given} \
             were given",
            projection.dynamic_coefficient_count()
        ));
        let phases = map.residues.len();
        comptime!(assert!(
            phases == projection.physical_rank(),
            "ProjectionInKernel: the projection has {} physical axes but {phases} residues were given",
            projection.physical_rank()
        ));
        let digits = map.digit_starts.len();
        comptime!(assert!(
            digits == projection.digit_dimensions().len(),
            "ProjectionInKernel: the projection reads digits of {} dimensions but {digits} digit \
             starts were given",
            projection.digit_dimensions().len()
        ));
        let radices = map.digit_radices.len();
        comptime!(assert!(
            radices == projection.digit_radix_count(),
            "ProjectionInKernel: the projection's digits read {} radices off the buffer but \
             {radices} were given",
            projection.digit_radix_count()
        ));
        ProjectionInKernel {
            shape,
            map,
            space,
            projection,
            width,
        }
    }
}

/// A coefficient in its physical axis's units: lines on the innermost, which must divide.
/// A backward coefficient is its two's complement, so a wrapping product and sum subtract it.
fn line_scale(
    space: &Space,
    projection: &Projection,
    width: usize,
    pa: usize,
    dimension: Dimension,
    scale: isize,
) -> u32 {
    let innermost = space.dimension_at(space.rank() - 1);
    if pa != projection.physical_rank() - 1 || width == 1 || dimension == innermost {
        return scale as u32;
    }
    assert!(
        scale.unsigned_abs().is_multiple_of(width),
        "ProjectionInKernel: {dimension:?} steps {scale} values of a physical axis read {width} at a time, \
         so one line spans more than one {dimension:?}; serve narrower lines, or give {dimension:?} a \
         physical axis of its own"
    );
    (scale / width as isize) as u32
}

#[cube]
impl ProjectionInKernel {
    /// Where a walk along `walk` from `pos` starts.
    ///
    /// # Assumptions
    ///
    /// - `pos` lies in the logical box.
    // `#[unroll]` needs a range loop over the comptime lists.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn anchor(&self, pos: CoordsDyn, #[comptime] walk: &WalkUnits) -> Anchor {
        let floors = comptime!(self.projection.floors_reevaluated(walk));
        let origins = comptime!(self.projection.digits_reevaluated(walk));
        let mut words = Coords::<u32>::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            words.push(self.anchored_axis(&pos, pa, walk));
        }
        #[unroll]
        for k in 0..floors.len() {
            words.push(self.floor_numerator(&pos, comptime!(floors[k])));
        }
        #[unroll]
        for j in 0..origins.len() {
            words.push(pos[comptime!(self.space.position(origins[j]))]);
        }

        Anchor { words }
    }

    /// The words an [`Anchor`] of `walk` holds, for a caller that stores it.
    pub(crate) fn anchor_len(&self, #[comptime] walk: &WalkUnits) -> comptime_type!(usize) {
        comptime!(self.projection.anchor_len(walk))
    }

    /// The source coordinate a walk from `anchor` lands on after `counts[i]` counts along each
    /// dimension `i` of `walk`.
    ///
    /// # Assumptions
    ///
    /// - `anchor` is this map's [`anchor`](Self::anchor) of `walk`.
    /// - `counts` holds one count per dimension of `walk`, in its order, and the position the
    ///   walk lands on lies in the logical box.
    ///
    /// # Guarantees
    ///
    /// - The result is [`to_source_pos`](Layout::to_source_pos) of the anchor's position moved
    ///   `counts[i]·coords_per_count` along each walked dimension.
    /// - Where every term the walk moves adds or holds ([`TermStep`]), it is the anchor's words
    ///   plus a product of each count by a constant, with no division and no remainder.
    pub(crate) fn advance(
        &self,
        anchor: &Anchor,
        counts: &Coords<u32>,
        #[comptime] walk: &WalkUnits,
    ) -> CoordsDyn {
        let walked = counts.len();
        comptime!(assert!(
            walked == walk.dimensions.len(),
            "ProjectionInKernel::advance: {walked} counts for a walk along {} dimensions",
            walk.dimensions.len()
        ));
        let floors = comptime!(self.projection.floors_reevaluated(walk));
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            let axis_map = comptime!(self.projection.physical_axis(pa).clone());
            let floored = comptime!(floors.iter().position(|&f| f == pa));
            let mut steps = Coords::<u32>::new();
            steps.push(anchor.words.at(pa));
            #[unroll]
            for t in 0..axis_map.terms().len() {
                let term = comptime!(axis_map.terms()[t]);
                if comptime!(walk.position(term.dimension).is_some()) {
                    steps.push(self.term_moved(anchor, counts, pa, t, walk));
                }
            }
            if comptime!(floored.is_some()) {
                steps.push(self.floor_moved(anchor, counts, pa, walk));
            }
            let n_steps = steps.len();
            out.push(steps.sum(comptime!((0..n_steps).collect::<Vec<_>>())));
        }

        out
    }

    /// The count of each dimension of `walk` that reaches `pos` from where that dimension's
    /// coordinate is zero: the counts of a walk of unit steps anchored there.
    pub(crate) fn counts_at(&self, pos: &CoordsDyn, #[comptime] walk: &WalkUnits) -> Coords<u32> {
        let mut counts = Coords::<u32>::new();
        #[unroll]
        for i in 0..walk.dimensions.len() {
            let walked = comptime!(walk.dimensions[i]);
            comptime!(assert!(
                walked.coords_per_count == 1,
                "ProjectionInKernel::counts_at: {:?} is walked {} coordinates a count, so a \
                 coordinate is not its count",
                walked.dimension,
                walked.coords_per_count
            ));
            counts.push(pos[comptime!(self.space.position(walked.dimension))]);
        }
        counts
    }

    /// Physical axis `pa` at `pos`, without the terms `walk` evaluates again: a digit read again
    /// where the walk lands, and a floor taken again, whose numerator is kept on its own.
    fn anchored_axis(
        &self,
        pos: &CoordsDyn,
        #[comptime] pa: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa).clone());
        let floor_again = comptime!(axis_map.reevaluates_floor(walk));
        // Exact steps stay outside the numerator, added after the floor.
        let mut offsets = Coords::<u32>::new();
        #[unroll]
        for t in 0..axis_map.terms().len() {
            let term = comptime!(axis_map.terms()[t]);
            let again = comptime!(
                term.digit.is_some()
                    && walk
                        .coords_per_count(term.dimension)
                        .is_some_and(|unit| axis_map.term_step(t, unit) == TermStep::Reevaluates)
            );
            let outside =
                comptime!(!axis_map.is_rational() || axis_map.static_offset_step(t).is_some());
            if comptime!(!again && outside) {
                offsets.push(self.term_at(pos, pa, t));
            }
        }
        let n_exact = offsets.len();
        let offset = offsets.sum(comptime!((0..n_exact).collect::<Vec<_>>()));

        if comptime!(!axis_map.is_rational() || floor_again) {
            offset
        } else {
            self.floored(self.floor_numerator(pos, pa), pa).plus(offset)
        }
    }

    /// The numerator of rational physical axis `pa`'s floor at `pos`: its residue plus every term
    /// kept inside the floor.
    fn floor_numerator(&self, pos: &CoordsDyn, #[comptime] pa: usize) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa).clone());
        let mut terms = Coords::<u32>::new();
        terms.push(self.map.residues.at(pa));
        #[unroll]
        for t in 0..axis_map.terms().len() {
            if comptime!(axis_map.static_offset_step(t).is_none()) {
                terms.push(self.term_at(pos, pa, t));
            }
        }
        let n_kept = terms.len();
        terms.sum(comptime!((0..n_kept).collect::<Vec<_>>()))
    }

    /// `numerator` divided by rational physical axis `pa`'s divisor, floored.
    fn floored(&self, numerator: u32, #[comptime] pa: usize) -> u32 {
        match comptime!(self.projection.divisor(pa)) {
            Divisor::Static(d) => numerator.divided_by(comptime!(d as u32)),
            Divisor::Dynamic { .. } => numerator.divided_by(self.map.coefficients.at(comptime!(
                self.projection.dynamic_divisor_index(pa).unwrap()
            ))),
        }
    }

    /// Term `t` of physical axis `pa` at `pos`: its digit of the coordinate, or the coordinate
    /// times its coefficient where it is summed ([`coefficient`](Self::coefficient)).
    fn term_at(&self, pos: &CoordsDyn, #[comptime] pa: usize, #[comptime] t: usize) -> u32 {
        let term = comptime!(self.projection.physical_axis(pa).terms()[t]);
        let coordinate = pos[comptime!(self.space.position(term.dimension))];
        if comptime!(term.digit.is_some()) {
            self.digit_term(coordinate, pa, t)
        } else {
            coordinate.times(self.coefficient(pa, t))
        }
    }

    /// The coefficient of plain term `t` of physical axis `pa` where it is summed, in the axis's
    /// units: past the floor a static step exact under the divisor, else its own, lined on the
    /// innermost axis ([`line_scale`]); a dynamic one read off the runtime map. What a coordinate
    /// of the term is multiplied by, at the anchor and along a walk alike.
    fn coefficient(&self, #[comptime] pa: usize, #[comptime] t: usize) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa).clone());
        let term = comptime!(axis_map.terms()[t]);
        let exact = comptime!(axis_map.static_offset_step(t));
        match (term.scale, exact) {
            (_, Some(step)) => comptime!(line_scale(
                &self.space,
                &self.projection,
                self.width,
                pa,
                term.dimension,
                step as isize
            ))
            .runtime(),
            (Scale::Static(s), None) => comptime!(line_scale(
                &self.space,
                &self.projection,
                self.width,
                pa,
                term.dimension,
                s
            ))
            .runtime(),
            (Scale::Dynamic { .. }, None) => self.map.coefficients.at(comptime!(
                self.projection.dynamic_scale_index(pa, t).unwrap()
            )),
        }
    }

    /// Term `t` of physical axis `pa`, a digit, at `coordinate` along its dimension: its digit of
    /// the coordinate plus the tile's start along the dimension, times its coefficient.
    fn digit_term(&self, coordinate: u32, #[comptime] pa: usize, #[comptime] t: usize) -> u32 {
        let term = comptime!(self.projection.physical_axis(pa).terms()[t]);
        let digit = comptime!(term.digit.unwrap());
        let (place_at, extent_at) = comptime!(self.projection.digit_radix_index(pa));
        let start = self
            .map
            .digit_starts
            .at(comptime!(self.projection.digit_start_index(term.dimension)));
        let quotient = coordinate
            .plus(start)
            .divided_by(self.radix(digit.place_value, place_at));
        let place = match digit.quotient {
            Quotient::Whole => quotient,
            Quotient::Modulo => quotient.remainder(self.radix(digit.extent, extent_at)),
        };
        place.times(comptime!(term.scale.get() as u32))
    }

    /// `radix`'s value: its constant, or the runtime map's digit radix `index`.
    fn radix(&self, #[comptime] radix: Radix, #[comptime] index: Option<usize>) -> u32 {
        match radix {
            Radix::Static(value) => comptime!(value as u32).runtime(),
            Radix::Dynamic => self.map.digit_radices.at(comptime!(index.unwrap())),
        }
    }

    /// What term `t` of physical axis `pa`, on a dimension of `walk`, adds to the anchor's word
    /// after `counts`: its constant step per count, or its digit read again where the walk lands.
    /// A term inside a floor taken again adds through the numerator ([`floor_moved`]).
    ///
    /// [`floor_moved`]: Self::floor_moved
    fn term_moved(
        &self,
        anchor: &Anchor,
        counts: &Coords<u32>,
        #[comptime] pa: usize,
        #[comptime] t: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa).clone());
        let term = comptime!(axis_map.terms()[t]);
        let i = comptime!(walk.position(term.dimension).unwrap());
        let unit = comptime!(walk.dimensions[i].coords_per_count);
        let count = counts.at(i);
        let inside_again =
            comptime!(axis_map.reevaluates_floor(walk) && axis_map.static_offset_step(t).is_none());
        match comptime!((
            axis_map.term_step(t, unit),
            inside_again,
            term.digit.is_some()
        )) {
            (_, true, _) | (TermStep::Holds, _, _) => 0u32.runtime(),
            (TermStep::Adds(c), false, true) => count.times(comptime!(c as u32)),
            (TermStep::Adds(c), false, false) => count.times(comptime!(line_scale(
                &self.space,
                &self.projection,
                self.width,
                pa,
                term.dimension,
                c
            ))),
            (TermStep::AddsCoefficient(coordinates), false, _) => count
                .times(comptime!(coordinates as u32))
                .times(self.coefficient(pa, t)),
            (TermStep::Reevaluates, false, _) => self.digit_term(
                self.walked_origin(anchor, term.dimension, walk)
                    .plus(count.times(comptime!(unit as u32))),
                pa,
                t,
            ),
        }
    }

    /// Rational physical axis `pa`'s floor taken again after `counts`: the anchor's numerator plus
    /// each walked term kept inside the floor, divided.
    fn floor_moved(
        &self,
        anchor: &Anchor,
        counts: &Coords<u32>,
        #[comptime] pa: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa).clone());
        let floors = comptime!(self.projection.floors_reevaluated(walk));
        let slot = comptime!(
            self.projection.physical_rank() + floors.iter().position(|&f| f == pa).unwrap()
        );
        let mut numerator = Coords::<u32>::new();
        numerator.push(anchor.words.at(slot));
        #[unroll]
        for t in 0..axis_map.terms().len() {
            let term = comptime!(axis_map.terms()[t]);
            let walked = comptime!(walk.position(term.dimension));
            if comptime!(walked.is_some() && axis_map.static_offset_step(t).is_none()) {
                let i = comptime!(walked.unwrap());
                numerator.push(
                    counts
                        .at(i)
                        .times(comptime!(walk.dimensions[i].coords_per_count as u32))
                        .times(self.coefficient(pa, t)),
                );
            }
        }
        let n = numerator.len();
        self.floored(numerator.sum(comptime!((0..n).collect::<Vec<_>>())), pa)
    }

    /// The anchor's coordinate along `dimension`, a dimension of `walk` some digit is evaluated
    /// again along.
    fn walked_origin(
        &self,
        anchor: &Anchor,
        #[comptime] dimension: Dimension,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let slot = comptime!(
            self.projection.physical_rank()
                + self.projection.floors_reevaluated(walk).len()
                + self
                    .projection
                    .digits_reevaluated(walk)
                    .iter()
                    .position(|&d| d == dimension)
                    .unwrap()
        );
        anchor.words.at(slot)
    }
}

#[cube]
impl Layout for ProjectionInKernel {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let none = comptime!(WalkUnits::none());
        self.advance(&self.anchor(pos, &none), &Coords::<u32>::new(), &none)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.to_dyn()
    }

    /// The logical box; physical validity is the [`Window`](crate::Window)'s question.
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.shape.within(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate cubecl as cubecl_runtime;
    use cubecl_runtime::TestRuntime;

    const A: Dimension = Dimension(0);
    const B: Dimension = Dimension(1);

    /// The innermost physical axis of `[A, B]` read two values a line, `⌊(8a + 3b)/2⌋`: `A`'s
    /// term exact past the floor, `B`'s inside it. The axis walked `count` counts of `unit` along
    /// `A` from `(a, b)`, and the map evaluated where that walk lands.
    #[cube(harness)]
    fn walked_on_a_divided_axis(a: u32, b: u32, count: u32, #[comptime] unit: usize) -> (u32, u32) {
        let projection = comptime!(Projection::new(
            &[A, B],
            &[
                PhysicalAxisMap::of(A),
                PhysicalAxisMap::affine(&[(A, 8), (B, 3)]).over(2),
            ],
        ));
        let space = comptime!(Space::new(&[(A, 8), (B, 8)]));
        let map = ProjectionInKernel::new(
            Coords::constant(comptime!(vec![8, 4])),
            RuntimeMap::integral(2usize),
            space,
            projection,
            2usize,
        );
        let walk = comptime!(WalkUnits::along(A, unit));
        let mut from = CoordsDyn::new();
        from.push(a);
        from.push(b);
        let mut counts = Coords::<u32>::new();
        counts.push(count);
        let walked = map.advance(&map.anchor(from, &walk), &counts, &walk);
        let mut landed = CoordsDyn::new();
        landed.push(a + count * comptime!(unit as u32));
        landed.push(b);
        (walked[1], map.to_source_pos(landed)[1])
    }

    /// A step taken past the floor is scaled to lines where the walk starts as where it lands.
    #[test]
    fn anchor_and_advance_agree_on_an_innermost_divided_axis() {
        for unit in [1, 2] {
            for (a, b, count) in
                (0..3).flat_map(|a| (0..4).flat_map(move |b| (0..3).map(move |n| (a, b, n))))
            {
                let (walked, landed) =
                    walked_on_a_divided_axis::harness::<TestRuntime>(a, b, count, unit);
                assert_eq!(walked, landed, "a {a}, b {b}, {count} counts of {unit}");
            }
        }
    }
}
