//! An operand's dimensions mapped onto its buffer's physical axes.

use crate::Addressed;
use cubecl::zspace::SmallVec;

use crate::layout::storage::pieces_named;
use crate::{
    AnchorSlots, AxesBuilder, Composition, Decline, Digit, Dimension, Divisor, Geometry, Offset,
    PhysicalAxisMap, Quotient, Radix, Space, TermStep, WalkUnits,
};

/// An operand's dimensions mapped onto its buffer's physical axes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Projection {
    /// One entry per physical axis, in buffer order.
    physical: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]>,
    /// The dimensions, in tile order; the last is the vectorized dimension.
    dimensions: SmallVec<[Dimension; Space::MAX_RANK]>,
}

impl Projection {
    /// One dimension per physical axis at coefficient `1`, in buffer order. A dimension named again
    /// is its next finer piece.
    pub fn direct(dimensions: &[Dimension]) -> Self {
        Projection::tiled(dimensions, &pieces_named(dimensions))
    }

    /// [`direct`](Projection::direct) over a space's own dimensions.
    pub(crate) fn direct_over(space: &crate::Space) -> Self {
        Projection::direct(&space.dimensions().collect::<Vec<_>>())
    }

    /// [`direct`](Projection::direct) storage-tiled: `pieces` state each buffer axis as the
    /// dimension it is a piece of and how many of that dimension's pieces are coarser.
    ///
    /// # Panics
    ///
    /// As [`new`](Projection::new).
    pub fn tiled(dimensions: &[Dimension], pieces: &[(Dimension, usize)]) -> Self {
        let physical = pieces
            .iter()
            .map(|&(dimension, piece)| PhysicalAxisMap::piece_of(dimension, piece))
            .collect();
        Projection::of_maps(dimensions, physical)
    }

    /// The same operand with each dimension's storage fragments merged back into one coordinate.
    pub(crate) fn untiled(&self) -> Projection {
        let carried = self.carried_groups();
        let physical: Vec<PhysicalAxisMap> = carried
            .iter()
            .map(|&pa| self.physical[pa].clone().whole())
            .collect();
        Projection::new(&self.dimensions, &physical)
    }

    /// How many coordinates address this operand: its physical axes with tiled fragments merged.
    pub fn coordinate_rank(&self) -> usize {
        self.carried_groups().len()
    }

    /// The same buffer addressed by physical position: physical axis `p` relabeled `Dimension(p)`.
    pub(crate) fn positional(&self) -> Projection {
        let carried = self.carried_groups();
        let dimensions: Vec<Dimension> = (0..carried.len()).map(|p| Dimension(p as u8)).collect();
        let physical: Vec<PhysicalAxisMap> = self
            .physical
            .iter()
            .enumerate()
            .map(|(pa, map)| {
                let at = match map.addressed() {
                    Addressed::Broadcast | Addressed::ByDigit => {
                        carried.iter().position(|&q| q == pa)
                    }
                    addressed => carried
                        .iter()
                        .position(|&q| self.physical[q].addressed() == addressed),
                }
                .expect("collected above");
                PhysicalAxisMap::piece_of(dimensions[at], map.piece())
            })
            .collect();
        Projection::new(&dimensions, &physical)
    }

    /// The first physical axis of each coordinate group, in buffer order.
    fn carried_groups(&self) -> Vec<usize> {
        assert!(
            self.is_invertible() || !self.is_tiled(),
            "Projection: an affine map cannot also be storage-tiled; its physical axes do not \
             group into coordinates"
        );
        let mut carried: Vec<usize> = Vec::new();
        for (pa, map) in self.physical.iter().enumerate() {
            match map.addressed() {
                // Still a buffer axis, or a digit's, so it is its own group.
                Addressed::Broadcast | Addressed::ByDigit => carried.push(pa),
                addressed => {
                    if !carried
                        .iter()
                        .any(|&q| self.physical[q].addressed() == addressed)
                    {
                        carried.push(pa);
                    }
                }
            }
        }
        carried
    }

    /// Whether some dimension is split across several physical fragments.
    pub(crate) fn is_tiled(&self) -> bool {
        self.dimensions
            .iter()
            .any(|&dimension| self.fragments(dimension) > 1)
    }

    /// How many physical axes `dimension` is split across. A broadcast dimension is spread over
    /// none, and one read as digits is read, not stored, across its axes: neither is a tiling of
    /// it, so each is one fragment, the same answer an untiled dimension gives.
    fn fragments(&self, dimension: Dimension) -> usize {
        match self.addresses(dimension) && !self.reads_digits_of(dimension) {
            false => 1,
            true => self.carriers(dimension).len(),
        }
    }

    /// Whether some physical axis reads a digit of `dimension` ([`folded`](Projection::folded)).
    fn reads_digits_of(&self, dimension: Dimension) -> bool {
        self.digit_dimensions().contains(&dimension)
    }

    /// The dimensions some physical axis reads a digit of, in axis order, each once: the order of
    /// a `RuntimeMap`'s digit starts.
    pub(crate) fn digit_dimensions(&self) -> SmallVec<[Dimension; Space::MAX_RANK]> {
        let mut dimensions = SmallVec::new();
        for dimension in self.physical.iter().filter_map(|m| m.digit_dimension()) {
            if !dimensions.contains(&dimension) {
                dimensions.push(dimension);
            }
        }
        dimensions
    }

    /// Where `dimension`'s start sits in a `RuntimeMap`'s digit starts.
    pub(crate) fn digit_start_index(&self, dimension: Dimension) -> usize {
        self.digit_dimensions()
            .iter()
            .position(|&d| d == dimension)
            .expect("Projection::digit_start_index: no axis reads a digit of this dimension")
    }

    /// Merges the dimensions in `fold`, outermost first, each with its extent, into one dimension
    /// `into`: `folded(P, &[(OH, oh), (OW, ow)])` gives `P = oh_index * ow + ow_index`. On an
    /// operand whose axes are each one dimension, each folded axis becomes a stored piece of
    /// `into`, sized by the buffer, whose shape must match `fold`. On a gathered operand each term
    /// on a folded dimension reads that
    /// dimension's digit of `into`. Call it with the same `fold` on every operand that has the
    /// folded dimensions.
    pub fn folded(&self, into: Dimension, fold: &[(Dimension, usize)]) -> Projection {
        assert!(
            fold.iter().all(|&(d, _)| self.dimensions.contains(&d)),
            "Projection::folded: {fold:?} names a dimension this operand does not have, {:?}",
            self.dimensions
        );
        let positions: Vec<usize> = fold.iter().map(|&(d, _)| self.position(d)).collect();
        assert!(
            fold.len() > 1 && positions.windows(2).all(|pair| pair[1] == pair[0] + 1),
            "Projection::folded: fold two or more adjacent dimensions of this operand, each once, \
             outermost first; got {fold:?} of {:?}",
            self.dimensions
        );
        assert!(
            !self.dimensions.contains(&into),
            "Projection::folded: {into:?} is already one of this operand's dimensions"
        );
        assert!(
            self.is_invertible() || !self.is_tiled(),
            "Projection::folded: a storage-tiled operand is not folded; its pieces would read as \
             separate digits"
        );
        let is_folded = |d: Dimension| fold.iter().any(|&(f, _)| f == d);
        let first = positions[0];
        let dimensions: Vec<Dimension> = self
            .dimensions
            .iter()
            .enumerate()
            .filter_map(|(at, &d)| match (at == first, is_folded(d)) {
                (true, _) => Some(into),
                (false, true) => None,
                (false, false) => Some(d),
            })
            .collect();
        let physical: Vec<PhysicalAxisMap> = match self.is_invertible() {
            true => self
                .physical
                .iter()
                .map(|map| match map.identity_dimension() {
                    Some(d) if is_folded(d) => PhysicalAxisMap::piece_of(
                        into,
                        fold.iter().position(|&(f, _)| f == d).unwrap(),
                    ),
                    _ => map.clone(),
                })
                .collect(),
            false => self
                .physical
                .iter()
                .map(|map| map.clone().folded(into, fold))
                .collect(),
        };
        Projection::new(&dimensions, &physical)
    }

    /// How physical axis `pa` carries `dimension`: the positions of its finer carriers, whose
    /// extents a coordinate is divided by first, and the one whose extent bounds `pa`'s coordinate
    /// (`None` for the outermost, which keeps the whole quotient).
    pub(crate) fn carried_by(
        &self,
        pa: usize,
        dimension: Dimension,
    ) -> (SmallVec<[usize; Space::MAX_RANK]>, Option<usize>) {
        assert!(
            !self.reads_digits_of(dimension),
            "Projection::carried_by: {dimension:?} is read as digits, not stored as pieces, so its \
             axes do not divide a coordinate by the buffer's extents"
        );
        let carriers = self.carriers(dimension);
        let at = carriers.iter().position(|&q| q == pa).unwrap_or_else(|| {
            panic!(
                "Projection::carried_by: physical axis {pa} does not carry {dimension:?} (carried \
                 by {carriers:?})"
            )
        });
        let finer = SmallVec::from_slice(&carriers[at + 1..]);
        (finer, (at > 0).then_some(pa))
    }

    /// The physical axes storing finer pieces of the coordinate physical axis `pa` stores a piece
    /// of, whose extents its place value is the product of; none for a coordinate stored whole.
    pub(crate) fn finer_pieces_of_axis(&self, pa: usize) -> impl Iterator<Item = usize> {
        let dimension = self.physical[pa].terms()[0].dimension;
        let (finer, _) = self.carried_by(pa, dimension);
        finer.into_iter()
    }

    /// Whether this operand addresses `dimension`; a spanned but unaddressed dimension is a
    /// broadcast.
    pub(crate) fn addresses(&self, dimension: Dimension) -> bool {
        self.physical.iter().any(|m| m.addresses(dimension))
    }

    /// The physical axes carrying `dimension`, its pieces coarsest first. Panics if there are none.
    pub(crate) fn carriers(&self, dimension: Dimension) -> SmallVec<[usize; Space::MAX_RANK]> {
        let mut carriers: SmallVec<[usize; Space::MAX_RANK]> = (0..self.physical.len())
            .filter(|&q| {
                self.physical[q]
                    .terms()
                    .iter()
                    .any(|t| t.dimension == dimension)
            })
            .collect();
        assert!(
            !carriers.is_empty(),
            "Projection::carriers: {dimension:?} addresses no physical axis of this operand"
        );
        carriers.sort_by_key(|&pa| self.physical[pa].piece());
        carriers
    }

    /// This operand's scales' projection, one scale per `block`.
    /// `block` must partition its axis ([`disjoint`](PhysicalAxisMap::disjoint)); panics otherwise.
    pub fn scales_per(&self, block: Dimension) -> Projection {
        let pa = self
            .physical
            .iter()
            .position(|map| map.terms().iter().any(|term| term.dimension == block))
            .unwrap_or_else(|| {
                panic!("Projection::scales_per: {block:?} addresses no axis of this operand")
            });
        let map = &self.physical[pa];
        let terms = map.terms();
        let at = terms
            .iter()
            .position(|term| term.dimension == block)
            .unwrap();
        assert!(
            map.composition() == Composition::Disjoint && at + 1 < terms.len(),
            "Projection::scales_per: {block:?} must partition its axis with a dimension inside it \
             (PhysicalAxisMap::disjoint(&[({block:?}, block), (inside, 1)]))"
        );
        let radix = terms[at].scale.bound();
        let kept: Vec<(Dimension, usize)> = terms[..=at]
            .iter()
            .map(|term| (term.dimension, term.scale.bound() / radix))
            .collect();
        let omitted: Vec<Dimension> = terms[at + 1..].iter().map(|term| term.dimension).collect();
        let mut physical = self.physical.clone();
        physical[pa] = match kept.as_slice() {
            [(dimension, 1)] => PhysicalAxisMap::of(*dimension),
            _ => PhysicalAxisMap::disjoint(&kept),
        };
        Projection {
            physical,
            dimensions: self
                .dimensions
                .iter()
                .copied()
                .filter(|dimension| !omitted.contains(dimension))
                .collect(),
        }
    }

    /// `dimensions` in the tile's logical order, `physical` one per physical axis in buffer order.
    ///
    /// # Panics
    ///
    /// Where the axes a dimension is stored as ([`PhysicalAxisMap::piece_of`]) do not state each
    /// of its pieces once.
    pub fn new(dimensions: &[Dimension], physical: &[PhysicalAxisMap]) -> Self {
        Projection::of_maps(dimensions, physical.iter().cloned().collect())
    }

    /// [`new`](Projection::new) over maps the caller built for it, taken as they are.
    pub(crate) fn of_maps(
        dimensions: &[Dimension],
        physical: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]>,
    ) -> Self {
        let projection = Projection {
            physical,
            dimensions: SmallVec::from_slice(dimensions),
        };
        projection.assert_pieces_stated_once();
        projection
    }

    fn assert_pieces_stated_once(&self) {
        for &dimension in &self.dimensions {
            let mut pieces: SmallVec<[usize; Space::MAX_RANK]> = self
                .physical
                .iter()
                .filter(|map| map.is_identity(dimension))
                .map(PhysicalAxisMap::piece)
                .collect();
            pieces.sort();
            assert!(
                pieces.iter().copied().eq(0..pieces.len()),
                "Projection: the axes {dimension:?} is stored as state its pieces {pieces:?}, not \
                 each of {} once",
                pieces.len()
            );
        }
    }

    /// Whether this is the [`direct`](Projection::direct) mapping.
    pub(crate) fn is_direct(&self) -> bool {
        self.physical.len() == self.dimensions.len()
            && self
                .physical
                .iter()
                .zip(self.dimensions.iter())
                .all(|(map, &dimension)| map.is_identity(dimension))
    }

    /// The dimensions carried by the trailing physical axes, up to the innermost broadcast.
    pub(crate) fn dense_labels(&self) -> Vec<Dimension> {
        let mut labels: Vec<Dimension> = self
            .physical
            .iter()
            .rev()
            .map_while(|map| match map.addressed() {
                Addressed::By(dimension) => Some(dimension),
                Addressed::ByDigit | Addressed::Broadcast => None,
            })
            .collect();
        labels.reverse();
        labels
    }

    pub fn physical_rank(&self) -> usize {
        self.physical.len()
    }

    pub(crate) fn logical_rank(&self) -> usize {
        self.dimensions.len()
    }

    pub(crate) fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// `dimension`'s index in the logical order, which is the order a coordinate comes in.
    pub fn position(&self, dimension: Dimension) -> usize {
        self.dimensions
            .iter()
            .position(|&a| a == dimension)
            .expect("Projection::position: dimension not spanned by this operand")
    }

    pub(crate) fn physical_axis(&self, pa: usize) -> &PhysicalAxisMap {
        &self.physical[pa]
    }

    /// `dimension`'s coefficient along physical axis `pa`, `0` when it does not address it.
    pub fn scale(&self, pa: usize, dimension: Dimension) -> isize {
        self.physical[pa].scale(dimension)
    }

    /// The offset along physical axis `pa`.
    pub fn offset(&self, pa: usize) -> Offset {
        self.physical[pa].offset()
    }

    /// Physical axis `pa`'s divisor, [`Static(1)`](Divisor::Static) for integer mappings.
    pub(crate) fn divisor(&self, pa: usize) -> Divisor {
        self.physical[pa].divisor()
    }

    /// Whether any physical axis is [rational](PhysicalAxisMap::is_rational).
    pub(crate) fn is_rational(&self) -> bool {
        self.physical.iter().any(|m| m.is_rational())
    }

    /// Whether any physical axis has a negative or dynamic offset, or a backward coefficient.
    pub(crate) fn may_underflow(&self) -> bool {
        self.has_negative_scale()
            || self.physical.iter().any(|m| match m.offset() {
                Offset::Static(o) => o < 0,
                Offset::Dynamic => true,
            })
    }

    /// Whether any physical axis steps backward along a dimension.
    pub(crate) fn has_negative_scale(&self) -> bool {
        self.physical.iter().any(|m| m.has_negative_scale())
    }

    /// How term `t` of physical axis `pa` moves along `walk`, read `width` values a line
    /// ([`PhysicalAxisMap::term_step`]); `None` where the walk does not move its dimension.
    pub(crate) fn walked_step(
        &self,
        pa: usize,
        t: usize,
        walk: &WalkUnits,
        width: usize,
    ) -> Option<TermStep> {
        let axis = &self.physical[pa];
        let dimension = axis.terms()[t].dimension;
        walk.coords_per_count(dimension)
            .map(|unit| axis.term_step(t, unit, self.lines_of_term(pa, dimension, width)))
    }

    /// The values a line holds where a term of `dimension` on physical axis `pa` is read `width`
    /// values a line: `width` on the innermost axis for a dimension other than the innermost,
    /// whose coordinate already counts lines; `1` everywhere else.
    pub(crate) fn lines_of_term(&self, pa: usize, dimension: Dimension, width: usize) -> usize {
        let innermost = self.dimensions.last() == Some(&dimension);
        match pa + 1 == self.physical_rank() && !innermost {
            true => width,
            false => 1,
        }
    }

    /// Whether the ends of a run along `walk`, read `width` values a line, bound every position
    /// between them: no coefficient steps backward, and every term the walk moves adds or holds
    /// ([`TermStep`]), where a term evaluated again, a digit wrapping or a floor, may land outside
    /// both ends' box.
    pub(crate) fn run_bounded_by_ends(&self, walk: &WalkUnits, width: usize) -> bool {
        !self.has_negative_scale()
            && (0..self.physical_rank()).all(|pa| {
                (0..self.physical[pa].terms().len())
                    .all(|t| self.walked_step(pa, t, walk, width) != Some(TermStep::Reevaluates))
            })
    }

    /// Where each word of a walk's [`Anchor`](crate::Anchor) over this map sits, the walk read
    /// `width` values a line.
    pub(crate) fn anchor_slots(&self, walk: &WalkUnits, width: usize) -> AnchorSlots {
        let reevaluated = |pa: usize, t: usize| {
            self.walked_step(pa, t, walk, width) == Some(TermStep::Reevaluates)
        };
        let floors = (0..self.physical_rank())
            .filter(|&pa| {
                let axis = &self.physical[pa];
                (0..axis.terms().len()).any(|t| axis.kept_in_floor(t) && reevaluated(pa, t))
            })
            .collect();
        let origins = walk
            .dimensions
            .iter()
            .map(|walked| walked.dimension)
            .filter(|&dimension| {
                (0..self.physical_rank()).any(|pa| {
                    self.physical[pa]
                        .terms()
                        .iter()
                        .enumerate()
                        .any(|(t, term)| {
                            term.dimension == dimension
                                && term.digit.is_some()
                                && reevaluated(pa, t)
                        })
                })
            })
            .collect();
        AnchorSlots {
            axes: self.physical_rank(),
            floors,
            origins,
        }
    }

    /// This operand's buffer axes read as the digits of the coordinates they store pieces of: the
    /// map `ProjectionInKernel` evaluates for a storage-tiled buffer, so a read lands on its pieces
    /// with no division by the buffer's extents left below it.
    ///
    /// # Assumptions
    ///
    /// - `self` is the [`untiled`](Projection::untiled) map of a storage-tiled operand, reading
    ///   no digit.
    /// - `positional` is the same operand's [`positional`](Projection::positional) map.
    /// - `radices` holds each buffer axis's place value and extent, in loads, in buffer order:
    ///   [`Radix::Static`] where the operand states its tiles, [`Radix::Dynamic`] where the buffer
    ///   sets them.
    ///
    /// # Guarantees
    ///
    /// - One physical axis per buffer axis, in buffer order.
    /// - An axis storing a piece of a coordinate is one term on that coordinate's dimension, at
    ///   coefficient `1`: the whole quotient for its coarsest piece, each finer one modulo its
    ///   extent. A coordinate stored whole is its own coarsest piece, of place value `1`, so it
    ///   too reads its start. A broadcast axis keeps its map.
    /// - Evaluated at a coordinate plus its start, an axis is the piece the buffer stores that
    ///   coordinate's value at.
    pub(crate) fn pieces_as_digits(
        &self,
        positional: &Projection,
        radices: &[(Radix, Radix)],
    ) -> Projection {
        assert!(
            self.digit_dimensions().is_empty(),
            "Projection::pieces_as_digits: the untiled map should read no digit; a folded operand is \
             never storage-tiled, so its pieces cannot be read as digits too"
        );
        assert!(
            radices.len() == positional.physical_rank(),
            "Projection::pieces_as_digits: there should be one radix pair a buffer axis, and there \
             are {} for {}",
            radices.len(),
            positional.physical_rank()
        );
        let physical: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]> = (0..positional
            .physical_rank())
            .map(|pa| {
                let coordinate = positional.physical[pa].terms()[0].dimension;
                let stored = &self.physical[coordinate.0 as usize];
                let (place_value, extent) = radices[pa];
                let (_, modulo) = positional.carried_by(pa, coordinate);
                match stored.terms().len() {
                    0 => stored.clone(),
                    _ => PhysicalAxisMap::digit_of(
                        stored.terms()[0].dimension,
                        Digit {
                            place_value,
                            extent,
                            quotient: modulo.map_or(Quotient::Whole, |_| Quotient::Modulo),
                        },
                    ),
                }
            })
            .collect();
        Projection::of_maps(&self.dimensions, physical)
    }

    /// The radices this map's digits read off the buffer, each as its physical axis and whether
    /// it is that axis's extent rather than its place value, in the order a `RuntimeMap`'s digit
    /// radices hold them: axis major, the place value first. A coarsest digit's extent is never
    /// read, so it takes no slot.
    pub(crate) fn dynamic_radices(&self) -> impl Iterator<Item = (usize, bool)> + '_ {
        self.physical.iter().enumerate().flat_map(|(pa, axis)| {
            let digit = axis.terms().iter().find_map(|term| term.digit);
            let place = digit.is_some_and(|d| d.place_value == Radix::Dynamic);
            let extent =
                digit.is_some_and(|d| d.quotient == Quotient::Modulo && d.extent == Radix::Dynamic);
            [(pa, false, place), (pa, true, extent)]
                .into_iter()
                .filter_map(|(pa, is_extent, read)| read.then_some((pa, is_extent)))
        })
    }

    /// Where physical axis `pa`'s place value and extent sit among the
    /// [`dynamic_radices`](Projection::dynamic_radices), `None` for one read as a constant.
    pub(crate) fn digit_radix_index_of_axis(&self, pa: usize) -> (Option<usize>, Option<usize>) {
        let index = |is_extent| {
            self.dynamic_radices()
                .position(|radix| radix == (pa, is_extent))
        };
        (index(false), index(true))
    }

    /// Refuses a backward coefficient for `what`, which takes the window's physical box as its
    /// bound: such a window reaches below its origin cell.
    pub(crate) fn refuse_backward(&self, what: &str) {
        assert!(
            !self.has_negative_scale(),
            "{what}: a backward coefficient reaches below its window's physical box; stage the \
             operand in a tiled stage (`StageStorage::Tiled`)"
        );
    }

    /// How many elements of physical axis `pa` a region covers: `1 + Σ (extent - 1) * |scale|`.
    /// Rational and dynamic dimensions report the widest span their bounds admit.
    pub(crate) fn span(&self, pa: usize, extent_of: impl Fn(Dimension) -> usize) -> usize {
        let map = &self.physical[pa];
        let field: usize = map
            .terms()
            .iter()
            .map(|t| t.largest_coordinate(extent_of(t.dimension)) * t.scale.bound())
            .sum();
        1 + field.div_ceil(map.divisor().bound())
    }

    /// `Disjoint` if every physical axis is partitioned by its dimensions, `Translated` if every
    /// one is, some moved by an offset, else `Overlapping`.
    pub(crate) fn composition(&self) -> Composition {
        let one_cell_a_position = |m: &PhysicalAxisMap| {
            m.composition() != Composition::Overlapping && m.divisor().is_unit()
        };
        if !self.physical.iter().all(one_cell_a_position) {
            return Composition::Overlapping;
        }
        match self
            .physical
            .iter()
            .any(|m| m.composition() == Composition::Translated)
        {
            true => Composition::Translated,
            false => Composition::Disjoint,
        }
    }

    /// Panics if a disjoint dimension's coefficients are not the products of its finer extents.
    pub(crate) fn validate_composition(&self, extent_of: impl Fn(Dimension) -> usize) {
        for (pa, map) in self.physical.iter().enumerate() {
            let radices = map.claimed_radices();
            if radices.is_empty() {
                continue;
            }
            assert!(
                map.offset() == Offset::Static(0),
                "Projection: physical axis {pa} claims to be partitioned by its dimensions, but carries \
                 the offset {:?}; a partition starts at 0",
                map.offset()
            );
            // Only extents below the coarsest are read, keeping a `Dynamic` top dimension out.
            let mut expected = 1;
            for (i, (dimension, coefficient)) in radices.iter().enumerate().rev() {
                assert!(
                    *coefficient == expected,
                    "Projection: physical axis {pa} claims its dimensions partition it, so {dimension:?} \
                     must step by {expected} (the extents below it); it steps by {coefficient}"
                );
                if i == 0 {
                    break;
                }
                expected *= extent_of(*dimension);
            }
        }
    }

    pub(crate) fn is_invertible(&self) -> bool {
        self.physical
            .iter()
            .all(|m| m.identity_dimension().is_some())
    }

    /// Why a projection stated over a plain buffer, where a dimension spans several axes, cannot
    /// be served at `vector_size`, if it cannot: its dimensions must run in the order of its
    /// coordinates, each dimension's at its first axis, and a line wider than one value runs
    /// along the last of them, so the innermost axis must carry the last dimension. (A storage
    /// tiling states its own order; its loads are its tiles.)
    pub(crate) fn dimensions_in_axis_order(&self, vector_size: usize) -> Result<(), Decline> {
        if !self.is_tiled() {
            return Ok(());
        }
        let by_axis: Vec<Dimension> = self
            .carried_groups()
            .into_iter()
            .filter_map(|pa| match self.physical[pa].addressed() {
                Addressed::By(dimension) => Some(dimension),
                Addressed::ByDigit | Addressed::Broadcast => None,
            })
            .collect();
        let stated: Vec<Dimension> = self
            .dimensions
            .iter()
            .copied()
            .filter(|&dimension| self.addresses(dimension))
            .collect();
        let innermost = self.physical.last().map(PhysicalAxisMap::addressed);
        let lined = vector_size == 1 || innermost == stated.last().copied().map(Addressed::By);
        match stated == by_axis && lined {
            true => Ok(()),
            false => Err(Decline::DimensionsOutOfAxisOrder { stated, by_axis }),
        }
    }

    /// Why a gathered projection cannot be served at `vector_size`, if it cannot.
    pub(crate) fn validate(&self, vector_size: usize) -> Result<(), Decline> {
        if self.is_invertible() {
            return Ok(());
        }
        if self.physical.is_empty() || self.dimensions.is_empty() {
            return Err(Decline::NoCoordinate);
        }
        // The innermost physical axis is addressed in lines, so it must step by one element.
        if vector_size > 1 {
            let innermost = self.dimensions[self.dimensions.len() - 1];
            let last = &self.physical[self.physical.len() - 1];
            let steps_by_lines = match last.composition() {
                Composition::Disjoint => {
                    last.terms().last().map(|t| t.dimension) == Some(innermost)
                }
                Composition::Overlapping | Composition::Translated => last.is_identity(innermost),
            };
            if !steps_by_lines {
                return Err(Decline::GatherInnermostNotInLines);
            }
        }
        for &dimension in self.dimensions.iter() {
            self.validate_gathered_dimension(dimension)?;
        }
        Ok(())
    }

    /// A gathered operand must be untiled: each addressed dimension maps to one physical axis,
    /// unless every axis it addresses reads a digit of it.
    fn validate_gathered_dimension(&self, dimension: Dimension) -> Result<(), Decline> {
        let mut addressing = self.physical.iter().filter(|m| m.addresses(dimension));
        let count = addressing.clone().count();
        match count < 2 || addressing.all(|m| m.digit_dimension() == Some(dimension)) {
            true => Ok(()),
            false => Err(Decline::GatherDimensionAddressedTwice(dimension)),
        }
    }

    /// Start writing a projection one buffer axis at a time, in buffer order ([`AxesBuilder`]).
    pub fn axes() -> AxesBuilder {
        AxesBuilder::new()
    }
}

// Queries against a buffer's runtime `Geometry`.
impl Projection {
    /// Whether the buffer's axes can be addressed without two positions landing on one cell.
    pub fn is_addressable(&self, geometry: &Geometry) -> bool {
        if self.composition() == Composition::Overlapping {
            return true;
        }
        geometry.addressing_axes_by_stride().windows(2).all(|pair| {
            let (_, (_, stride)) = pair[0];
            let (_, (finer_extent, finer_stride)) = pair[1];
            stride >= finer_extent * finer_stride
        })
    }
}

// Indices into the runtime coefficient (unsigned) and offset (signed) arrays: physical axis
// major, term order within, each dimension's divisor last.
impl Projection {
    /// Term `t` of physical axis `pa` in the runtime coefficient array, `None` if static.
    pub(crate) fn dynamic_scale_index(&self, pa: usize, t: usize) -> Option<usize> {
        if !self.physical_axis(pa).terms()[t].scale.is_dynamic() {
            return None;
        }
        let within = self.physical_axis(pa).terms()[..t]
            .iter()
            .filter(|term| term.scale.is_dynamic())
            .count();
        Some(self.coefficient_base(pa) + within)
    }

    /// Physical axis `pa`'s divisor in the runtime coefficient array, `None` if static.
    pub(crate) fn dynamic_divisor_index(&self, pa: usize) -> Option<usize> {
        if !self.physical_axis(pa).divisor().is_dynamic() {
            return None;
        }
        Some(self.coefficient_base(pa) + self.physical_axis(pa).dynamic_scale_count())
    }

    /// Where physical axis `pa`'s entries start in the runtime coefficient array.
    fn coefficient_base(&self, pa: usize) -> usize {
        (0..pa)
            .map(|i| self.physical_axis(i))
            .map(|m| m.dynamic_scale_count() + m.divisor().is_dynamic() as usize)
            .sum()
    }

    /// The length of the runtime coefficient array.
    pub(crate) fn dynamic_coefficient_count(&self) -> usize {
        self.coefficient_base(self.physical_rank())
    }

    /// Physical axis `pa`'s offset in the runtime offset array, `None` if static.
    pub(crate) fn dynamic_offset_index(&self, pa: usize) -> Option<usize> {
        if !self.physical_axis(pa).offset().is_dynamic() {
            return None;
        }
        Some(
            (0..pa)
                .map(|i| self.physical_axis(i))
                .filter(|m| m.offset().is_dynamic())
                .count(),
        )
    }

    /// The length of the runtime offset array.
    pub(crate) fn dynamic_offset_count(&self) -> usize {
        self.physical
            .iter()
            .filter(|m| m.offset().is_dynamic())
            .count()
    }

    /// Whether any coefficient is only known at runtime.
    pub(crate) fn has_dynamic_scales(&self) -> bool {
        self.physical.iter().any(|m| m.has_dynamic_scale())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scale;

    /// Block scales omit the in-block dimension and count the split axis in blocks.
    #[test]
    fn scales_per_block_omit_the_dimension_inside_it() {
        let values = Projection::new(
            &[M, KB, KI],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::disjoint(&[(KB, 32), (KI, 1)]),
            ],
        );
        let scales = values.scales_per(KB);
        assert_eq!(scales.dimensions(), &[M, KB]);
        assert_eq!(scales.physical_rank(), 2);
        assert!(scales.physical[1].is_identity(KB));
    }

    /// Scales per a middle dimension keep the coarser dimension, rescaled to count blocks.
    #[test]
    fn scales_per_a_middle_dimension_keep_what_is_above_it() {
        let values = Projection::new(
            &[M, KO, KB, KI],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::disjoint(&[(KO, 64), (KB, 32), (KI, 1)]),
            ],
        );
        let per_block = values.scales_per(KB);
        assert_eq!(per_block.dimensions(), &[M, KO, KB]);
        assert_eq!(per_block.scale(1, KO), 2);
        assert_eq!(per_block.scale(1, KB), 1);
        let per_outer = values.scales_per(KO);
        assert_eq!(per_outer.dimensions(), &[M, KO]);
        assert!(per_outer.physical[1].is_identity(KO));
    }

    #[test]
    #[should_panic(expected = "must partition its axis with a dimension inside it")]
    fn scales_per_an_unsplit_dimension_are_refused() {
        Projection::direct(&[M, KB]).scales_per(KB);
    }

    const A: Dimension = Dimension(0);
    const B: Dimension = Dimension(1);
    const R: Dimension = Dimension(2);

    const M: Dimension = Dimension(3);
    const KB: Dimension = Dimension(4);
    const KI: Dimension = Dimension(5);
    const KO: Dimension = Dimension(6);

    #[test]
    fn direct_is_direct() {
        let p = Projection::direct(&[A, B]);
        assert!(p.is_direct());
        assert_eq!(p.physical_rank(), 2);
        assert_eq!(p.dimensions(), &[A, B]);
    }

    #[test]
    fn direct_span_is_the_edge() {
        let p = Projection::direct(&[A, B]);
        assert_eq!(p.span(0, |_| 8), 8);
    }

    #[test]
    fn affine_span_is_the_receptive_field() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_direct());
        assert_eq!(p.scale(0, A), 2);
        assert_eq!(p.scale(0, R), 3);
        assert_eq!(p.scale(0, B), 0);
        // 1 + (4-1)*2 + (3-1)*3 = 13
        assert_eq!(p.span(0, |a| if a == A { 4 } else { 3 }), 13);
        let q = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 1), (R, 1)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(q.span(0, |a| if a == A { 4 } else { 1 }), 4);
    }

    #[test]
    fn innermost_axis_rides_no_coarser_physical_axis() {
        let refused = Projection::new(
            &[A, B],
            &[
                PhysicalAxisMap::affine(&[(A, 1), (B, 2)]),
                PhysicalAxisMap::of(B),
            ],
        )
        .validate(4);
        assert_eq!(refused, Err(Decline::GatherDimensionAddressedTwice(B)));
    }

    const OH: Dimension = Dimension(7);
    const OW: Dimension = Dimension(8);
    const RW: Dimension = Dimension(9);

    /// A 2-D input `[h, w, c]` read at `h = oh * 2 + r - 1`, `w = ow + rw - 1`.
    fn input() -> Projection {
        Projection::new(
            &[OH, OW, R, RW, B],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (R, 1)]).shifted(-1),
                PhysicalAxisMap::affine(&[(OW, 1), (RW, 1)]).shifted(-1),
                PhysicalAxisMap::of(B),
            ],
        )
    }

    /// Folded into `A = oh * 7 + ow`, the rows and columns stay coordinates of their own and
    /// nothing is storage-tiled.
    #[test]
    fn a_folded_gather_reads_its_digits_on_their_own_axes() {
        let p = input().folded(A, &[(OH, 5), (OW, 7)]);
        assert_eq!(p.dimensions(), &[A, R, RW, B]);
        assert_eq!(p.validate(4), Ok(()));
        assert!(!p.is_tiled());
        assert_eq!(&p.digit_dimensions()[..], &[A]);
        assert_eq!(p.coordinate_rank(), 3);
        // 1 + 4 * 2 + 2 * 1, whatever the region's extent along `A`.
        assert_eq!(p.span(0, |d| if d == A { 16 } else { 3 }), 11);
    }

    /// A direct operand's fold is pieces of the one dimension, its radices the buffer's.
    #[test]
    fn a_folded_direct_operand_stores_pieces() {
        let p = Projection::direct(&[OH, OW, B]).folded(A, &[(OH, 5), (OW, 7)]);
        assert_eq!(p, Projection::tiled(&[A, B], &[(A, 0), (A, 1), (B, 0)]));
    }

    #[test]
    fn innermost_must_be_identity() {
        let refused = Projection::new(
            &[A, R],
            &[PhysicalAxisMap::of(A), PhysicalAxisMap::affine(&[(R, 2)])],
        )
        .validate(4);
        assert_eq!(refused, Err(Decline::GatherInnermostNotInLines));
    }

    #[test]
    fn innermost_may_gather_when_scalar() {
        Projection::new(
            &[A, R],
            &[PhysicalAxisMap::of(A), PhysicalAxisMap::affine(&[(R, 2)])],
        )
        .validate(1)
        .unwrap();
    }

    #[test]
    fn a_gather_still_rejects_tiled_storage_when_scalar() {
        let refused = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
                PhysicalAxisMap::piece_of(B, 1),
            ],
        )
        .validate(1);
        assert_eq!(refused, Err(Decline::GatherDimensionAddressedTwice(B)));
    }

    #[test]
    fn a_gather_rejects_tiled_storage() {
        let refused = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
                PhysicalAxisMap::piece_of(B, 1),
            ],
        )
        .validate(4);
        assert_eq!(refused, Err(Decline::GatherDimensionAddressedTwice(B)));
    }

    #[test]
    fn a_dimension_addressing_nothing_is_a_broadcast() {
        let p = Projection::new(
            &[A, R, B],
            &[PhysicalAxisMap::affine(&[(A, 2)]), PhysicalAxisMap::of(B)],
        );
        p.validate(4).unwrap();
        assert!(!p.addresses(R));
        assert!(p.addresses(A) && p.addresses(B));
    }

    #[test]
    fn tiled_is_not_a_gather() {
        let p = Projection::tiled(&[A, B], &[(A, 0), (B, 0), (A, 1), (B, 1)]);
        assert!(!p.is_direct());
        assert!(p.is_tiled());
        assert!(p.is_invertible());
        p.validate(4).unwrap();
    }

    #[test]
    fn a_gather_is_not_tiled() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_tiled());
        p.validate(4).unwrap();
    }

    #[test]
    fn a_tiled_spec_matches_its_buffer() {
        use crate::TileSpec;

        const BATCH: Dimension = Dimension(3);
        let spec = TileSpec::new(Projection::new(
            &[BATCH, A, B],
            &[(BATCH, 0), (A, 0), (B, 0), (A, 1), (B, 1)]
                .map(|(dimension, piece)| PhysicalAxisMap::piece_of(dimension, piece)),
        ));

        assert_eq!(spec.projection.physical_rank(), 5);
        let [p0, p1, p2] = [Dimension(0), Dimension(1), Dimension(2)];
        assert_eq!(
            spec.projection.positional(),
            Projection::tiled(
                &[p0, p1, p2],
                &[(p0, 0), (p1, 0), (p2, 0), (p1, 1), (p2, 1)]
            )
        );
        assert_eq!(
            Projection::tiled(
                &[BATCH, A, B],
                &[(BATCH, 0), (A, 0), (B, 0), (A, 1), (B, 1)]
            ),
            spec.projection
        );
    }

    #[test]
    fn positional_of_a_plain_operand_is_the_identity() {
        assert_eq!(
            Projection::direct(&[B, A]).positional(),
            Projection::direct(&[Dimension(0), Dimension(1)])
        );
    }

    #[test]
    fn positional_of_a_gather_drops_the_affine_terms() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(
            p.positional(),
            Projection::direct(&[Dimension(0), Dimension(1)])
        );
    }

    #[test]
    fn a_ragged_tiling_orders_level_major() {
        let p = Projection::tiled(&[A, B], &[(A, 0), (B, 0), (A, 1), (B, 1), (A, 2)]);
        assert_eq!(p.physical_rank(), 5);
        // `[A0, B0, A1, B1, A2]`.
        assert_eq!(p.carriers(A).as_slice(), &[0, 2, 4]);
        assert_eq!(p.carriers(B).as_slice(), &[1, 3]);
        assert_eq!(p.carried_by(0, A), (SmallVec::from_slice(&[2, 4]), None));
        assert_eq!(p.carried_by(2, A), (SmallVec::from_slice(&[4]), Some(2)));
        assert_eq!(p.carried_by(4, A), (SmallVec::new(), Some(4)));
        assert_eq!(p.carried_by(1, B), (SmallVec::from_slice(&[3]), None));
        assert!(p.is_invertible());
        p.validate(4).unwrap();
    }

    /// Weighing each carrier's coordinate by the extents finer than it gives back the coordinate.
    #[test]
    fn carried_coordinates_weigh_back_to_the_coordinate() {
        let shape = [3usize, 8];
        let p = Projection::tiled(&[A], &[(A, 0), (A, 1)]);
        let block = |pa| {
            p.carried_by(pa, A)
                .0
                .iter()
                .map(|&q| shape[q])
                .product::<usize>()
        };

        for coord in [0usize, 1, 7, 8, 9, 19, 23] {
            let physical = |pa| match p.carried_by(pa, A).1 {
                Some(m) => (coord / block(pa)) % shape[m],
                None => coord / block(pa),
            };
            assert_eq!(physical(0) * block(0) + physical(1) * block(1), coord);
        }
    }

    /// `[m, k]` stored `[m/16, k/16, m%8, k/2%4, k/8%2, m/8%2, k%2]`, as a tensor core takes a tile
    /// of sixteen by sixteen.
    fn tensor_core() -> Projection {
        Projection::tiled(
            &[A, B],
            &[(A, 0), (B, 0), (A, 2), (B, 2), (B, 1), (A, 1), (B, 3)],
        )
    }

    /// The FP8 activation stored for RDNA4's matrix instruction, `[m/16][k/8][m%16][k%8]`, read
    /// eight values a load, so its buffer counts `[m/16][k/8][16][1]` loads: its tiles stated, its
    /// tile counts the buffer's.
    fn fp8_tiles() -> (Projection, [(Radix, Radix); 4]) {
        let tiled = Projection::tiled(&[A, B], &[(A, 0), (B, 0), (A, 1), (B, 1)]);
        let radices = [
            (Radix::Static(16), Radix::Dynamic),
            (Radix::Static(1), Radix::Dynamic),
            (Radix::Static(1), Radix::Static(16)),
            (Radix::Static(1), Radix::Static(1)),
        ];
        (tiled, radices)
    }

    /// A digit's value at `x`, its radices static.
    fn digit_at(digit: Digit, x: usize) -> usize {
        let place = match digit.place_value {
            Radix::Static(place) => place,
            Radix::Dynamic => unreachable!("a digit read on the host has a static place"),
        };
        match (digit.quotient, digit.extent) {
            (Quotient::Whole, _) => x / place,
            (Quotient::Modulo, Radix::Static(extent)) => x / place % extent,
            (Quotient::Modulo, Radix::Dynamic) => unreachable!("a finer piece's extent is stated"),
        }
    }

    /// Each piece of `[m/16][k/8][m%16][k%8]` reads as the digit `carried_by` splits off its
    /// coordinate, at every coordinate of a buffer `[3][4][16][1]` loads.
    #[test]
    fn pieces_of_a_tiled_operand_read_as_its_digits() {
        let (tiled, radices) = fp8_tiles();
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &radices);
        let positional = tiled.positional();
        let shape = [3, 4, 16, 1];
        assert_eq!(digits.dimensions(), &[A, B]);
        assert_eq!(digits.physical_rank(), 4);
        assert_eq!(&digits.digit_dimensions()[..], &[A, B]);
        for pa in 0..4 {
            let term = digits.physical_axis(pa).terms()[0];
            let digit = term.digit.expect("every axis should store a piece");
            let c = Dimension(term.dimension.0);
            let (finer, modulo) = positional.carried_by(pa, c);
            assert_eq!(term.scale, Scale::Static(1));
            for x in 0..[48, 4][c.0 as usize] {
                let stripped = x / finer.iter().map(|&q| shape[q]).product::<usize>();
                let piece = modulo.map_or(stripped, |m| stripped % shape[m]);
                assert_eq!(digit_at(digit, x), piece, "axis {pa} at {x}");
            }
        }
    }

    /// `K` stored whole beside `M`'s two pieces reads as a coarsest digit of place one, so a read
    /// adds the window's start along it as along every other coordinate.
    #[test]
    fn a_coordinate_stored_whole_reads_as_a_digit_of_place_one() {
        let tiled = Projection::tiled(&[A, B], &[(A, 0), (B, 0), (A, 1)]);
        let radices = [
            (Radix::Static(8), Radix::Dynamic),
            (Radix::Static(1), Radix::Dynamic),
            (Radix::Static(1), Radix::Static(8)),
        ];
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &radices);
        assert_eq!(
            digits.physical_axis(1).terms()[0].digit,
            Some(Digit {
                place_value: Radix::Static(1),
                extent: Radix::Dynamic,
                quotient: Quotient::Whole,
            })
        );
        assert_eq!(&digits.digit_dimensions()[..], &[A, B]);
    }

    /// An axis whose radices the buffer sets reads them off the runtime map, each axis's place
    /// value before its extent, axis after axis; a coarsest piece's extent takes no slot.
    #[test]
    fn unstated_piece_extents_read_as_dynamic_radices() {
        let (tiled, _) = fp8_tiles();
        let unstated = [(Radix::Dynamic, Radix::Dynamic); 4];
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &unstated);
        assert_eq!(
            digits.dynamic_radices().collect::<Vec<_>>(),
            vec![
                (0, false),
                (1, false),
                (2, false),
                (2, true),
                (3, false),
                (3, true)
            ]
        );
        assert_eq!(digits.digit_radix_index_of_axis(3), (Some(4), Some(5)));
    }

    /// Every radix of `[m/16][k/8][m%16][k%8]` the buffer does not set is stated, so none is read.
    #[test]
    fn stated_piece_extents_read_no_radix() {
        let (tiled, stated) = fp8_tiles();
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &stated);
        assert_eq!(digits.dynamic_radices().count(), 0);
    }

    #[test]
    #[should_panic(expected = "should read no digit")]
    fn pieces_as_digits_refuses_a_projection_read_as_digits() {
        let folded = input().folded(A, &[(OH, 5), (OW, 7)]);
        let positional = Projection::direct(&[A, B]);
        folded.pieces_as_digits(&positional, &[(Radix::Static(1), Radix::Static(1)); 2]);
    }

    /// Walked a row at a time, the row inside a tile is evaluated again, which wraps.
    #[test]
    fn a_walk_reevaluating_a_modulo_digit_is_not_bounded_by_its_ends() {
        let (tiled, radices) = fp8_tiles();
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &radices);
        assert!(!digits.run_bounded_by_ends(&WalkUnits::along(A, 1), 1));
    }

    /// Walked a tile of rows at a time, the tile adds one and the row inside it holds.
    #[test]
    fn a_walk_of_whole_tiles_is_bounded_by_its_ends() {
        let (tiled, radices) = fp8_tiles();
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &radices);
        assert!(digits.run_bounded_by_ends(&WalkUnits::along(A, 16), 1));
    }

    /// The row inside a tile, evaluated again along a walk of rows, reads where the walk started:
    /// the anchor keeps that row past its axes' words.
    #[test]
    fn an_anchor_keeps_the_start_of_a_dimension_whose_digit_is_evaluated_again() {
        let (tiled, radices) = fp8_tiles();
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &radices);
        let slots = digits.anchor_slots(&WalkUnits::along(A, 1), 1);
        assert_eq!(slots.origins, vec![A]);
        assert_eq!(slots.origin_slot(A), 4);
    }

    /// Two loads along `k` at a time add to the tile of `k` and hold inside it: the anchor is its
    /// axes' words alone.
    #[test]
    fn an_anchor_moved_only_by_constants_holds_its_axes_alone() {
        let (tiled, radices) = fp8_tiles();
        let digits = tiled
            .untiled()
            .pieces_as_digits(&tiled.positional(), &radices);
        assert_eq!(
            digits.anchor_slots(&WalkUnits::along(B, 2), 1).word_count(),
            4
        );
    }

    /// Which piece an axis is comes from its statement, not from the axes around it.
    #[test]
    fn a_piece_is_carried_as_stated_wherever_its_axis_lies() {
        let p = tensor_core();
        assert_eq!(p.carried_by(0, A), (SmallVec::from_slice(&[5, 2]), None));
        assert_eq!(p.carried_by(5, A), (SmallVec::from_slice(&[2]), Some(5)));
        assert_eq!(p.carried_by(2, A), (SmallVec::new(), Some(2)));
        assert_eq!(p.carried_by(4, B), (SmallVec::from_slice(&[3, 6]), Some(4)));
    }

    /// The buffer addressed by position keeps the pieces its axes state.
    #[test]
    fn positional_keeps_the_pieces_stated() {
        assert_eq!(tensor_core().positional(), tensor_core());
    }

    /// The axes of one dimension merge into one, whichever piece the first of them is.
    #[test]
    fn untiled_merges_the_pieces_stated() {
        let p = Projection::tiled(&[A, B], &[(A, 1), (B, 0), (A, 0)]);
        assert_eq!(p.untiled(), Projection::direct(&[A, B]));
    }

    #[test]
    #[should_panic(expected = "not each of 2 once")]
    fn a_piece_stated_twice_is_refused() {
        Projection::tiled(&[A, B], &[(A, 0), (B, 0), (A, 0)]);
    }

    #[test]
    fn is_invertible_false_for_a_stencil_map() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_invertible());
        assert!(Projection::direct(&[A, B]).is_invertible());

        let with_offset = Projection::new(
            &[A, B],
            &[
                PhysicalAxisMap::affine(&[(A, 1)]).shifted(-1),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!with_offset.is_invertible());
        assert_eq!(with_offset.offset(0), Offset::Static(-1));
        assert_eq!(with_offset.offset(1), Offset::Static(0));
    }

    /// A backward coefficient lowers the lowest cell, so a region's span is the whole width
    /// between its ends and the origin may land below the buffer.
    #[test]
    fn a_backward_coefficient_widens_the_span_and_may_underflow() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::scaled(&[(A, Scale::Static(1)), (R, Scale::Static(-2))])
                    .shifted(1),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(p.scale(0, R), -2);
        assert!(p.has_negative_scale());
        assert!(p.may_underflow());
        let extent = |a: Dimension| if a == A { 4 } else { 3 };
        assert_eq!(p.span(0, extent), 1 + 3 + 4);
        assert!(!Projection::direct(&[A, B]).has_negative_scale());
    }

    #[test]
    fn may_underflow_tracks_negative_offsets_and_scales() {
        let padded = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 1)]).shifted(-1),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(padded.may_underflow());

        let shifted = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 1)]).shifted(1),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!shifted.may_underflow());
        assert!(!Projection::direct(&[A, B]).may_underflow());

        let dynamic_offset = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 1)]).shifted(Offset::Dynamic),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(dynamic_offset.may_underflow());

        let backward = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::scaled(&[(A, Scale::Static(1)), (R, Scale::Static(-2))]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(backward.may_underflow());
    }

    /// Physical axis major, term order within; offsets indexed apart from coefficients.
    #[test]
    fn dynamic_terms_index_in_projection_order() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::scaled(&[
                    (A, Scale::Dynamic { max: 2 }),
                    (R, Scale::Dynamic { max: 2 }),
                ]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(p.dynamic_coefficient_count(), 2);
        assert_eq!(p.dynamic_offset_count(), 0);
        assert_eq!(p.dynamic_scale_index(0, 0), Some(0));
        assert_eq!(p.dynamic_scale_index(0, 1), Some(1));
        assert_eq!(p.dynamic_offset_index(0), None);
        assert_eq!(p.dynamic_scale_index(1, 0), None);

        let mixed = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::scaled(&[(A, Scale::Static(2)), (R, Scale::Dynamic { max: 2 })]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(mixed.dynamic_coefficient_count(), 1);
        assert_eq!(mixed.dynamic_scale_index(0, 0), None);
        assert_eq!(mixed.dynamic_scale_index(0, 1), Some(0));
        assert_eq!(mixed.dynamic_offset_index(0), None);

        let with_dynamic_offset = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 2 }), (R, Scale::Static(1))])
                    .shifted(Offset::Dynamic),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(with_dynamic_offset.dynamic_coefficient_count(), 1);
        assert_eq!(with_dynamic_offset.dynamic_offset_count(), 1);
        assert_eq!(with_dynamic_offset.dynamic_scale_index(0, 0), Some(0));
        assert_eq!(with_dynamic_offset.dynamic_scale_index(0, 1), None);
        assert_eq!(with_dynamic_offset.dynamic_offset_index(0), Some(0));
        assert_eq!(with_dynamic_offset.dynamic_offset_index(1), None);

        let two_offsets = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 1)]).shifted(Offset::Dynamic),
                PhysicalAxisMap::scaled(&[(B, Scale::Dynamic { max: 2 })]).shifted(Offset::Dynamic),
            ],
        );
        assert_eq!(two_offsets.dynamic_offset_index(0), Some(0));
        assert_eq!(two_offsets.dynamic_offset_index(1), Some(1));
        assert_eq!(two_offsets.dynamic_scale_index(1, 0), Some(0));
    }

    #[test]
    fn a_dynamic_coefficient_is_not_invertible() {
        let p = Projection::new(
            &[A, B],
            &[
                PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 2 })]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_invertible());
        p.validate(4).unwrap();
    }

    #[test]
    fn rational_projection_properties() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 100), (R, 133)])
                    .shifted(-50)
                    .over(133),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(p.is_rational());
        assert_eq!(p.divisor(0), Divisor::Static(133));
        assert_eq!(p.divisor(1), Divisor::Static(1));
        assert_eq!(p.span(1, |_| 8), 8);
    }

    #[test]
    fn rational_dimension_conservative_span() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 100), (R, 133)])
                    .shifted(-50)
                    .over(133),
                PhysicalAxisMap::of(B),
            ],
        );
        // field = 300, span = 1 + ⌈300 / 133⌉ = 4
        assert_eq!(p.span(0, |a| if a == A { 4 } else { 1 }), 4);
        // field = 433, span = 1 + ⌈433 / 133⌉ = 5
        assert_eq!(p.span(0, |a| if a == A { 4 } else { 2 }), 5);
    }

    #[test]
    fn a_dynamic_divisor_spans_against_its_min() {
        let bounded = |min| {
            Projection::new(
                &[A, R, B],
                &[
                    PhysicalAxisMap::affine(&[(A, 100), (R, 133)])
                        .shifted(-50)
                        .over(Divisor::Dynamic { min }),
                    PhysicalAxisMap::of(B),
                ],
            )
        };
        // field = (4-1)*100 + (1-1)*133 = 300, so span = 1 + ⌈300 / min⌉.
        let extents = |a| if a == A { 4 } else { 1 };
        assert_eq!(bounded(133).span(0, extents), 4);
        assert_eq!(bounded(100).span(0, extents), 4);
        assert_eq!(bounded(50).span(0, extents), 7);
        let exact = |d| {
            Projection::new(
                &[A, R, B],
                &[
                    PhysicalAxisMap::affine(&[(A, 100), (R, 133)])
                        .shifted(-50)
                        .over(d),
                    PhysicalAxisMap::of(B),
                ],
            )
            .span(0, extents)
        };
        for d in 50..160 {
            assert!(exact(d) <= bounded(50).span(0, extents));
        }
    }

    #[test]
    fn a_dynamic_coefficient_spans_against_its_max() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 3 }), (R, Scale::Static(1))]),
                PhysicalAxisMap::of(B),
            ],
        );
        // field = (4-1)*3 + (2-1)*1 = 10, so span = 11.
        assert_eq!(p.span(0, |a| if a == A { 4 } else { 2 }), 11);
        let at_max = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 3), (R, 1)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(at_max.span(0, |a| if a == A { 4 } else { 2 }), 11);
    }

    #[test]
    fn dynamic_divisor_indexing() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 2 }), (R, Scale::Static(1))])
                    .over(Divisor::Dynamic { min: 4 }),
                PhysicalAxisMap::scaled(&[(B, Scale::Dynamic { max: 2 })]),
            ],
        );
        assert_eq!(p.dynamic_coefficient_count(), 3);
        assert_eq!(p.dynamic_scale_index(0, 0), Some(0));
        assert_eq!(p.dynamic_scale_index(0, 1), None);
        assert_eq!(p.dynamic_divisor_index(0), Some(1));
        assert_eq!(p.dynamic_scale_index(1, 0), Some(2));
    }

    #[test]
    fn a_partition_of_higher_logical_rank_is_still_dense() {
        let blocked = Projection::new(
            &[A, B, R],
            &[
                PhysicalAxisMap::of(A),
                PhysicalAxisMap::disjoint(&[(B, 32), (R, 1)]),
            ],
        );
        assert_eq!(blocked.composition(), Composition::Disjoint);
        assert!(!blocked.is_direct());

        let gathered = Projection::new(
            &[A, B, R],
            &[
                PhysicalAxisMap::of(A),
                PhysicalAxisMap::affine(&[(B, 32), (R, 1)]),
            ],
        );
        assert_eq!(gathered.composition(), Composition::Overlapping);
    }

    #[test]
    fn a_partition_whose_radices_match_the_extents_is_accepted() {
        let extents = |dimension| match dimension {
            x if x == B => 4,
            x if x == R => 32,
            _ => 8,
        };
        Projection::new(
            &[A, B, R],
            &[
                PhysicalAxisMap::of(A),
                PhysicalAxisMap::disjoint(&[(B, 32), (R, 1)]),
            ],
        )
        .validate_composition(extents);
    }

    #[test]
    #[should_panic(expected = "must step by 32")]
    fn a_partition_whose_radices_contradict_the_extents_is_refused() {
        let extents = |dimension| match dimension {
            x if x == B => 4,
            x if x == R => 32,
            _ => 8,
        };
        Projection::new(
            &[A, B, R],
            &[
                PhysicalAxisMap::of(A),
                PhysicalAxisMap::disjoint(&[(B, 16), (R, 1)]),
            ],
        )
        .validate_composition(extents);
    }
}

/// The questions a launch asks once it holds the buffer's real extents and strides.
#[cfg(test)]
mod geometry_tests {
    use super::*;

    const K: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const B: Dimension = Dimension(2);
    const H: Dimension = Dimension(3);
    const S: Dimension = Dimension(4);
    const D: Dimension = Dimension(5);
    const M: Dimension = Dimension(6);
    const C: Dimension = Dimension(7);

    fn geometry(axes: &[(usize, usize)]) -> Geometry {
        Geometry::new(axes)
    }

    #[test]
    fn padding_is_addressable() {
        let p = Projection::direct(&[K, N]);
        let padded = geometry(&[(4096, 11136), (11008, 1)]);

        assert!(p.is_addressable(&padded));
    }

    #[test]
    fn a_narrowed_row_aliases() {
        let p = Projection::direct(&[K, N]);
        let narrowed = geometry(&[(4096, 4096), (11008, 1)]);

        assert!(!p.is_addressable(&narrowed));
    }

    #[test]
    fn an_in_place_cache_is_addressable_padded_or_permuted() {
        let p = Projection::direct(&[B, H, S, D]);

        let padded = geometry(&[(2, 8 * 4096 * 136), (8, 4096 * 136), (4096, 136), (128, 1)]);
        assert!(p.is_addressable(&padded));

        let permuted = geometry(&[(2, 8 * 4096 * 128), (8, 128), (4096, 8 * 128), (128, 1)]);
        assert!(p.is_addressable(&permuted));
    }

    #[test]
    fn a_window_is_exempt_from_the_aliasing_check() {
        let p = Projection::axes()
            .axis(B)
            .axis(crate::layout::projection::axes_builder::stencil(&[(M, 2), (K, 1)]).pad(1))
            .axis(C)
            .build();
        let g = geometry(&[(8, 64 * 32), (64, 32), (32, 1)]);

        assert_eq!(p.composition(), Composition::Overlapping);
        assert!(p.is_addressable(&g));
    }
}
