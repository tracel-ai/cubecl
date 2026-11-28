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
/// position's constants to, and reads again the terms that move by none from.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Anchor {
    /// The words [`AnchorSlots`] places.
    words: Coords<u32>,
}

/// Where each word of a walk's [`Anchor`] sits ([`Projection::anchor_slots`]): each physical
/// axis's value at the start without the terms the walk evaluates again, then the numerator of
/// each floor it takes again, then the start's coordinate along each dimension it reads a digit
/// of again.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AnchorSlots {
    /// The physical axes, one word each, first.
    pub axes: usize,
    /// The physical axes whose floor the walk takes again, in axis order.
    pub floors: Vec<usize>,
    /// The walked dimensions some digit is evaluated again along, in the walk's order.
    pub origins: Vec<Dimension>,
}

impl AnchorSlots {
    /// The words an anchor of these slots holds.
    pub fn word_count(&self) -> usize {
        self.axes + self.floors.len() + self.origins.len()
    }

    /// The word holding the numerator of physical axis `pa`'s floor, `None` where the walk does
    /// not take it again.
    pub fn floor_slot_of_axis(&self, pa: usize) -> Option<usize> {
        self.floors
            .iter()
            .position(|&floor| floor == pa)
            .map(|at| self.axes + at)
    }

    /// The word holding the start's coordinate along `dimension`.
    ///
    /// # Panics
    ///
    /// Where no digit of `dimension` is evaluated again.
    pub fn origin_slot(&self, dimension: Dimension) -> usize {
        let at = self
            .origins
            .iter()
            .position(|&origin| origin == dimension)
            .expect("AnchorSlots: a dimension read again should have its start kept");
        self.axes + self.floors.len() + at
    }
}

#[cube]
impl Anchor {
    /// Store this anchor's words into `words` from `at` on, to read it back later with
    /// [`load_from`](Anchor::load_from).
    pub fn store_into(&self, words: &mut Array<u32>, at: usize) {
        #[unroll]
        for w in 0..self.words.len() {
            words[at + w] = self.words.at(w);
        }
    }

    /// The anchor `len` words long stored in `words` from `at` on
    /// ([`ProjectionInKernel::anchor_len`]).
    pub fn load_from(words: &Array<u32>, at: usize, #[comptime] len: usize) -> Anchor {
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
    pub fn new(
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
            "ProjectionInKernel: the projection reads digits of {} dimensions, and there should be \
             a start for each, but {digits} were given",
            projection.digit_dimensions().len()
        ));
        let radices = map.digit_radices.len();
        comptime!(assert!(
            radices == projection.dynamic_radices().count(),
            "ProjectionInKernel: the projection's digits read {} radices off the buffer, and there \
             should be one for each, but {radices} were given",
            projection.dynamic_radices().count()
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

/// A coefficient in its physical axis's units, read `lines` values a line
/// ([`Projection::lines_of_term`]). A backward coefficient is its two's complement, so a wrapping
/// product and sum subtract it.
fn line_scale(dimension: Dimension, scale: isize, lines: usize) -> u32 {
    assert!(
        scale.unsigned_abs().is_multiple_of(lines),
        "ProjectionInKernel: {dimension:?} steps {scale} values of a physical axis read {lines} at a \
         time, and a line should span one {dimension:?} at most; serve narrower lines, or give \
         {dimension:?} a physical axis of its own"
    );
    (scale / lines as isize) as u32
}

#[cube]
impl ProjectionInKernel {
    /// Where a walk along `walk` from `pos` starts.
    // `#[unroll]` needs a range loop over the comptime lists.
    #[allow(clippy::needless_range_loop)]
    pub fn anchor(&self, pos: CoordsDyn, #[comptime] walk: &WalkUnits) -> Anchor {
        let slots = comptime!(self.projection.anchor_slots(walk, self.width));
        let mut words = Coords::<u32>::new();

        #[unroll]
        for pa in 0..slots.axes {
            words.push(self.anchored_axis(&pos, pa, walk));
        }
        #[unroll]
        for k in 0..slots.floors.len() {
            words.push(self.floor_numerator(&pos, comptime!(slots.floors[k])));
        }
        #[unroll]
        for j in 0..slots.origins.len() {
            words.push(pos[comptime!(self.space.position(slots.origins[j]))]);
        }

        Anchor { words }
    }

    /// The words an [`Anchor`] of `walk` holds, for a caller that stores it.
    pub fn anchor_len(&self, #[comptime] walk: &WalkUnits) -> comptime_type!(usize) {
        comptime!(self.projection.anchor_slots(walk, self.width).word_count())
    }

    /// The source coordinate a walk from `anchor` lands on after `counts[i]` counts along each
    /// dimension `i` of `walk`.
    ///
    /// # Assumptions
    ///
    /// - `anchor` is this map's [`anchor`](Self::anchor) of `walk`.
    /// - `counts` holds one count per dimension of `walk`, in its order.
    ///
    /// # Guarantees
    ///
    /// - The result is [`to_source_pos`](Layout::to_source_pos) of the anchor's position moved
    ///   `counts[i]·coords_per_count` along each walked dimension.
    /// - Where every term the walk moves adds a compile-time constant or holds ([`TermStep`]), it
    ///   is the anchor's words plus a product of each count by a constant, with no division and
    ///   no remainder; a term adding its runtime coefficient adds a product by that coefficient.
    pub fn advance(
        &self,
        anchor: &Anchor,
        counts: &Coords<u32>,
        #[comptime] walk: &WalkUnits,
    ) -> CoordsDyn {
        let walked = counts.len();
        comptime!(assert!(
            walked == walk.dimensions.len(),
            "ProjectionInKernel::advance: there should be a count a walked dimension, and there \
             are {walked} for {}",
            walk.dimensions.len()
        ));
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            out.push(self.advanced_axis(anchor, counts, pa, walk));
        }

        out
    }

    /// Physical axis `pa` where the walk lands: the anchor's word, each term's move, and the
    /// floor taken again.
    fn advanced_axis(
        &self,
        anchor: &Anchor,
        counts: &Coords<u32>,
        #[comptime] pa: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let n = comptime!(self.projection.physical_axis(pa).terms().len());
        let mut steps = Coords::<u32>::new();
        steps.push(anchor.words.at(pa));
        #[unroll]
        for t in 0..n {
            steps.push(self.term_moved(anchor, counts, pa, t, walk));
        }
        steps.push(self.floor_moved(anchor, counts, pa, walk));
        let n_steps = steps.len();
        steps.sum(comptime!((0..n_steps).collect::<Vec<_>>()))
    }

    /// Physical axis `pa` at `pos`, without the terms `walk` evaluates again: a digit read again
    /// where the walk lands, and a floor taken again, whose numerator is kept on its own.
    fn anchored_axis(
        &self,
        pos: &CoordsDyn,
        #[comptime] pa: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let n = comptime!(self.projection.physical_axis(pa).terms().len());
        let mut parts = Coords::<u32>::new();
        #[unroll]
        for t in 0..n {
            parts.push(self.anchored_term(pos, pa, t, walk));
        }
        parts.push(self.anchored_floor(pos, pa, walk));
        let n_parts = parts.len();
        parts.sum(comptime!((0..n_parts).collect::<Vec<_>>()))
    }

    /// Term `t` of physical axis `pa` at `pos`, as the anchor keeps it outside the floor: `0` for
    /// a term kept in the floor, and for a digit `walk` evaluates again.
    fn anchored_term(
        &self,
        pos: &CoordsDyn,
        #[comptime] pa: usize,
        #[comptime] t: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let axis = comptime!(self.projection.physical_axis(pa).clone());
        let step = comptime!(self.projection.walked_step(pa, t, walk, self.width));
        let left_out = comptime!(
            axis.kept_in_floor(t)
                || (axis.terms()[t].digit.is_some() && step == Some(TermStep::Reevaluates))
        );
        match left_out {
            true => 0u32.runtime(),
            false => self.term_at(pos, pa, t),
        }
    }

    /// Physical axis `pa`'s floor at `pos`, where the anchor takes it: `0` for an integer axis,
    /// and for a floor `walk` takes again.
    fn anchored_floor(
        &self,
        pos: &CoordsDyn,
        #[comptime] pa: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let slots = comptime!(self.projection.anchor_slots(walk, self.width));
        let taken_here = comptime!(
            self.projection.physical_axis(pa).is_rational()
                && slots.floor_slot_of_axis(pa).is_none()
        );
        match taken_here {
            true => self.floored(self.floor_numerator(pos, pa), pa),
            false => 0u32.runtime(),
        }
    }

    /// The numerator of rational physical axis `pa`'s floor at `pos`: its residue plus every term
    /// kept inside the floor.
    fn floor_numerator(&self, pos: &CoordsDyn, #[comptime] pa: usize) -> u32 {
        let n = comptime!(self.projection.physical_axis(pa).terms().len());
        let mut terms = Coords::<u32>::new();
        terms.push(self.map.residues.at(pa));
        #[unroll]
        for t in 0..n {
            terms.push(self.numerator_term(pos, pa, t));
        }
        let n_kept = terms.len();
        terms.sum(comptime!((0..n_kept).collect::<Vec<_>>()))
    }

    /// Term `t` of physical axis `pa` at `pos` as its floor's numerator holds it, `0` for a term
    /// stepped past the floor.
    fn numerator_term(&self, pos: &CoordsDyn, #[comptime] pa: usize, #[comptime] t: usize) -> u32 {
        match comptime!(self.projection.physical_axis(pa).kept_in_floor(t)) {
            true => self.term_at(pos, pa, t),
            false => 0u32.runtime(),
        }
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
        match comptime!(term.digit.is_some()) {
            true => self.digit_term(coordinate, pa, t),
            false => coordinate.times(self.coefficient(pa, t)),
        }
    }

    /// The coefficient of plain term `t` of physical axis `pa` where it is summed, in the axis's
    /// units: past the floor a static step exact under the divisor, else its own, read in lines
    /// where [`Projection::lines_of_term`] says; a dynamic one read off the runtime map. What a
    /// coordinate of the term is multiplied by, at the anchor and along a walk alike.
    fn coefficient(&self, #[comptime] pa: usize, #[comptime] t: usize) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa).clone());
        let term = comptime!(axis_map.terms()[t]);
        let lines = comptime!(
            self.projection
                .lines_of_term(pa, term.dimension, self.width)
        );
        let exact = comptime!(axis_map.static_offset_step(t));
        match (term.scale, exact) {
            (_, Some(step)) => {
                comptime!(line_scale(term.dimension, step as isize, lines)).runtime()
            }
            (Scale::Static(s), None) => comptime!(line_scale(term.dimension, s, lines)).runtime(),
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
        let (place_at, extent_at) = comptime!(self.projection.digit_radix_index_of_axis(pa));
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

    /// What term `t` of physical axis `pa` adds to the anchor's word after `counts` of `walk`: its
    /// constant step per count, its runtime coefficient's, or its digit read again where the walk
    /// lands; `0` where the walk does not move it, and for a term inside a floor taken again,
    /// which moves through the numerator ([`floor_moved`]).
    ///
    /// A term that moves by no constant outside a floor is a digit: a plain term's coefficient is
    /// read in lines exactly ([`line_scale`]), so its step per count is a whole number of them.
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
        let term = comptime!(self.projection.physical_axis(pa).terms()[t]);
        let step = comptime!(self.projection.walked_step(pa, t, walk, self.width));
        let slots = comptime!(self.projection.anchor_slots(walk, self.width));
        let through_floor = comptime!(
            slots.floor_slot_of_axis(pa).is_some()
                && self.projection.physical_axis(pa).kept_in_floor(t)
        );
        let i = comptime!(walk.position(term.dimension));
        match comptime!((step, through_floor, term.digit.is_some())) {
            (None, _, _) | (Some(TermStep::Holds), _, _) | (_, true, _) => 0u32.runtime(),
            (Some(TermStep::Adds(c)), false, _) => {
                counts.at(comptime!(i.unwrap())).times(comptime!(c as u32))
            }
            (Some(TermStep::AddsCoefficient(coordinates)), false, _) => counts
                .at(comptime!(i.unwrap()))
                .times(comptime!(coordinates as u32))
                .times(self.coefficient(pa, t)),
            (Some(TermStep::Reevaluates), false, true) => self.digit_term(
                anchor
                    .words
                    .at(comptime!(slots.origin_slot(term.dimension)))
                    .plus(counts.at(comptime!(i.unwrap())).times(comptime!(
                        walk.dimensions[i.unwrap()].coords_per_count as u32
                    ))),
                pa,
                t,
            ),
            (Some(TermStep::Reevaluates), false, false) => comptime!(panic!(
                "ProjectionInKernel::advance: a plain term outside its floor should move by a \
                 constant, its coefficient read in whole lines"
            )),
        }
    }

    /// Rational physical axis `pa`'s floor taken again after `counts`: the anchor's numerator plus
    /// each walked term kept inside the floor, divided; `0` where the walk does not take it again.
    fn floor_moved(
        &self,
        anchor: &Anchor,
        counts: &Coords<u32>,
        #[comptime] pa: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let slots = comptime!(self.projection.anchor_slots(walk, self.width));
        match comptime!(slots.floor_slot_of_axis(pa)) {
            Some(slot) => self.floored(self.moved_numerator(anchor, counts, pa, slot, walk), pa),
            None => 0u32.runtime(),
        }
    }

    /// The numerator of physical axis `pa`'s floor after `counts`, the anchor's at word `slot`.
    fn moved_numerator(
        &self,
        anchor: &Anchor,
        counts: &Coords<u32>,
        #[comptime] pa: usize,
        #[comptime] slot: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let n = comptime!(self.projection.physical_axis(pa).terms().len());
        let mut numerator = Coords::<u32>::new();
        numerator.push(anchor.words.at(slot));
        #[unroll]
        for t in 0..n {
            numerator.push(self.numerator_moved(counts, pa, t, walk));
        }
        let n_terms = numerator.len();
        numerator.sum(comptime!((0..n_terms).collect::<Vec<_>>()))
    }

    /// What term `t` of physical axis `pa` adds to its floor's numerator after `counts`: its
    /// coefficient times the coordinates the walk moves it, `0` where it is stepped past the
    /// floor or not walked.
    fn numerator_moved(
        &self,
        counts: &Coords<u32>,
        #[comptime] pa: usize,
        #[comptime] t: usize,
        #[comptime] walk: &WalkUnits,
    ) -> u32 {
        let axis = comptime!(self.projection.physical_axis(pa).clone());
        let i = comptime!(walk.position(axis.terms()[t].dimension));
        match comptime!(i.filter(|_| axis.kept_in_floor(t))) {
            Some(i) => counts
                .at(i)
                .times(comptime!(walk.dimensions[i].coords_per_count as u32))
                .times(self.coefficient(pa, t)),
            None => 0u32.runtime(),
        }
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
    /// `A` from `(a, b)`.
    #[cube(harness)]
    fn walked_on_a_divided_axis(a: u32, b: u32, count: u32, #[comptime] unit: usize) -> u32 {
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
        map.advance(&map.anchor(from, &walk), &counts, &walk)[1]
    }

    /// A step taken past the floor lands where the map does, its coefficient read in lines:
    /// `⌊(8a + 3b)/2⌋` with `a`'s eight values four lines, `⌊(4a + 3b)/2⌋`.
    #[test]
    fn anchor_and_advance_agree_on_an_innermost_divided_axis() {
        for unit in [1, 2] {
            for (a, b, count) in
                (0..3).flat_map(|a| (0..4).flat_map(move |b| (0..3).map(move |n| (a, b, n))))
            {
                let walked = walked_on_a_divided_axis::harness::<TestRuntime>(a, b, count, unit);
                let landed = (4 * (a + count * unit as u32) + 3 * b) / 2;
                assert_eq!(walked, landed, "a {a}, b {b}, {count} counts of {unit}");
            }
        }
    }
}
