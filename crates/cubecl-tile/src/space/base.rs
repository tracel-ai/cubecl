//! The coordinate space a tile lives in.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::algebra::gcd;
use crate::{Dimension, Extent, Level, Shape};

/// Every dimension with its extent, in canonical order, plus the runtime sizes of dynamic
/// dimensions.
#[derive(CubeType, CubeLaunch, Clone, Debug)]
pub struct Space {
    #[cube(comptime)]
    pub(crate) shape: Shape,
    pub(crate) sizes: Sequence<usize>,
}

// Identity is the comptime shape only; the sizes are runtime, never a key.
impl PartialEq for Space {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}
impl Eq for Space {}
impl std::hash::Hash for Space {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
    }
}

/// The comptime reads of a runtime space.
impl SpaceExpand {
    /// The comptime shape of a runtime `Space`.
    pub(crate) fn comptime(&self) -> Space {
        Space::from_shape(self.shape.clone())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Space {
        self.comptime()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn dimension_at(&self, i: usize) -> Dimension {
        self.shape.dimension_at(i)
    }

    pub fn position(&self, dimension: Dimension) -> usize {
        self.shape.position(dimension)
    }

    pub fn contains(&self, dimension: Dimension) -> bool {
        self.shape.contains(dimension)
    }

    pub fn dimensions(&self) -> impl Iterator<Item = Dimension> + '_ {
        self.shape.dimensions()
    }

    pub fn extent(&self, dimension: Dimension) -> usize {
        self.shape.extent(dimension)
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.shape.extent_at(i)
    }

    pub fn is_dynamic(&self, dimension: Dimension) -> bool {
        self.shape.is_dynamic(dimension)
    }

    /// The listed dimensions with their extents, in the order listed.
    pub fn subspace(&self, dimensions: &[Dimension]) -> Space {
        Space::from_shape(self.shape.subspace(dimensions))
    }
}

impl Space {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Whether `other` is this space with some one-wide dimensions dropped, order and extents kept.
    pub(crate) fn narrows_to(&self, other: &Space) -> bool {
        self.dimensions()
            .filter(|&dimension| other.contains(dimension))
            .eq(other.dimensions())
            && self
                .dimensions()
                .last()
                .is_some_and(|dimension| other.contains(dimension))
            && self
                .dimensions()
                .all(|dimension| match other.contains(dimension) {
                    true => self.extent(dimension) == other.extent(dimension),
                    false => self.extent(dimension) == 1,
                })
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn dimension_at(&self, i: usize) -> Dimension {
        self.shape.dimension_at(i)
    }

    pub fn position(&self, dimension: Dimension) -> usize {
        self.shape.position(dimension)
    }

    pub fn contains(&self, dimension: Dimension) -> bool {
        self.shape.contains(dimension)
    }

    pub fn dimensions(&self) -> impl Iterator<Item = Dimension> + '_ {
        self.shape.dimensions()
    }

    pub(crate) fn extent_raw(&self, dimension: Dimension) -> Extent {
        self.shape.extent_raw(dimension)
    }

    /// The dimension's comptime size; panics on a `Dynamic` dimension.
    pub fn extent(&self, dimension: Dimension) -> usize {
        self.shape.extent(dimension)
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.shape.extent_at(i)
    }

    pub fn is_dynamic(&self, dimension: Dimension) -> bool {
        self.shape.is_dynamic(dimension)
    }

    /// The listed dimensions with their extents, in the order listed.
    pub fn subspace(&self, dimensions: &[Dimension]) -> Space {
        Space::from_shape(self.shape.subspace(dimensions))
    }

    /// This box listed in `held`'s order where `held` spans its dimensions longer than one, the
    /// one-wide ones it leaves out first, as it is otherwise: the box a plane-resident accumulator
    /// writes through, its fragments' rows along its own leading dimensions whatever order the
    /// memory it drains into lists them in.
    pub(crate) fn in_order_of(&self, held: &Space) -> Space {
        let one_wide =
            |dimension: &Dimension| matches!(self.extent_raw(*dimension), Extent::Static(1));
        let (shared, left_out): (Vec<_>, Vec<_>) = self
            .dimensions()
            .partition(|&dimension| held.contains(dimension));
        let spanned = shared.len() == held.rank() && left_out.iter().all(one_wide);
        match spanned {
            true => self.subspace(
                &left_out
                    .into_iter()
                    .chain(held.dimensions())
                    .collect::<Vec<_>>(),
            ),
            false => self.clone(),
        }
    }
}

#[cube]
impl Space {
    /// The runtime operation space: the comptime space plus the runtime `sizes` of its
    /// `Dynamic` dimensions (aligned to dimension order; empty when fully `Static`).
    pub(crate) fn with_sizes(#[comptime] space: &Space, sizes: Sequence<usize>) -> Space {
        Space {
            shape: space.shape.clone(),
            sizes,
        }
    }

    /// Dimension `p`'s tile count for a sub-tile `edge`; comptime for a `Static` dimension.
    pub fn count(&self, #[comptime] p: usize, #[comptime] edge: usize) -> usize {
        match comptime!(self.shape.extent_raw(self.shape.dimension_at(p))) {
            Extent::Static(n) => comptime!(n.div_ceil(edge)).runtime(),
            Extent::Dynamic => (*self.sizes.index(p)).div_ceil(edge),
        }
    }
}

impl Space {
    /// The most dimensions a space holds inline.
    pub const MAX_RANK: usize = 6;

    pub fn new(extents: &[(Dimension, usize)]) -> Self {
        let extents: Vec<_> = extents
            .iter()
            .map(|&(a, n)| (a, Extent::Static(n)))
            .collect();
        Space::from_extents(&extents)
    }

    /// Construct directly from [`Extent`]s.
    pub(crate) fn from_extents(extents: &[(Dimension, Extent)]) -> Self {
        Space::from_shape(Shape::new(extents))
    }

    /// A host space of `shape`, with no runtime sizes.
    pub(crate) fn from_shape(shape: Shape) -> Self {
        Space {
            shape,
            sizes: Sequence::new(),
        }
    }

    /// Flip the listed dimensions to `Dynamic`.
    pub fn with_dynamic(mut self, dimensions: &[Dimension]) -> Self {
        for dimension in dimensions {
            assert!(
                self.contains(*dimension),
                "Space::with_dynamic: {dimension:?} is not a dimension of this space"
            );
        }
        let entries: Vec<_> = self
            .dimensions()
            .map(|a| {
                let extent = if dimensions.contains(&a) {
                    Extent::Dynamic
                } else {
                    self.extent_raw(a)
                };
                (a, extent)
            })
            .collect();
        self.shape = Shape::new(&entries);
        self
    }

    /// Every dimension dynamic (see [`with_dynamic`](Space::with_dynamic)).
    pub fn all_dynamic(self) -> Self {
        let dimensions: Vec<_> = self.dimensions().collect();
        self.with_dynamic(&dimensions)
    }

    /// Every dimension is [`Static`](Extent::Static), so the walk is fully comptime.
    pub(crate) fn is_static(&self) -> bool {
        self.shape.is_static()
    }

    /// The static extents, in dimension order.
    pub fn extents(&self) -> Vec<(Dimension, usize)> {
        self.dimensions()
            .map(|dimension| (dimension, self.extent(dimension)))
            .collect()
    }

    /// The leaf `levels` reach below this space.
    pub(crate) fn leaf(&self, levels: &[Level]) -> Space {
        levels
            .iter()
            .fold(self.clone(), |space, level| level.child(&space))
    }

    /// The smallest space containing every `part`, dimensions in first-appearance order, shared
    /// dimensions broadcast-merged. E.g. `{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`.
    pub fn merge(parts: &[&Space]) -> Space {
        let mut entries: SmallVec<[(Dimension, Extent); Space::MAX_RANK]> = SmallVec::new();

        for part in parts {
            for dimension in part.dimensions() {
                let extent = part.extent_raw(dimension);
                match entries.iter_mut().find(|(a, _)| *a == dimension) {
                    Some(slot) => slot.1 = slot.1.broadcast(extent),
                    None => entries.push((dimension, extent)),
                }
            }
        }
        Space::from_shape(Shape::new(&entries))
    }

    /// How far this box reaches along `dimension`: one where it does not name it.
    ///
    /// # Panics
    ///
    /// On a dynamic extent.
    pub fn along(&self, dimension: Dimension) -> usize {
        match self.contains(dimension) {
            true => self.extent(dimension),
            false => 1,
        }
    }

    /// Whether this box divides `other`: along each of its dimensions, its extent divides
    /// `other`'s, one where `other` does not name the dimension.
    ///
    /// # Panics
    ///
    /// On a dynamic extent.
    pub fn divides(&self, other: &Space) -> bool {
        self.dimensions()
            .all(|dimension| other.along(dimension).is_multiple_of(self.along(dimension)))
    }

    /// How many of `tile` this box is long along `dimension`: `None` where it is no whole number
    /// of them.
    ///
    /// # Panics
    ///
    /// On a dynamic extent.
    pub fn count_along(&self, dimension: Dimension, tile: &Space) -> Option<usize> {
        let (whole, part) = (self.along(dimension), tile.along(dimension));
        (1..=whole).find(|count| part * count == whole)
    }

    /// How many of this box, along each dimension it or a divisor names, the smallest box made of
    /// it that every one of `divisors` divides holds.
    pub(crate) fn counts_whole_for(&self, divisors: &[Space]) -> Space {
        let mut counts: Vec<(Dimension, usize)> = Vec::new();
        for dimension in core::iter::once(self)
            .chain(divisors)
            .flat_map(Space::dimensions)
        {
            if counts.iter().all(|&(counted, _)| counted != dimension) {
                counts.push((dimension, self.count_whole_for(divisors, dimension)));
            }
        }
        Space::new(&counts)
    }

    /// The fewest of this box along `dimension` that every one of `divisors` divides.
    fn count_whole_for(&self, divisors: &[Space], dimension: Dimension) -> usize {
        (1..)
            .find(|count| {
                divisors.iter().all(|divisor| {
                    (self.along(dimension) * count).is_multiple_of(divisor.along(dimension))
                })
            })
            .expect("some count of a box should be a multiple of every divisor")
    }

    /// How many `tile`s this space holds, laid side by side along every dimension `tile` names; a
    /// dimension it does not name it spans whole, one tile serving every position along it. An
    /// `f16` activation loaded `{k 8}` fills a step `{k 16, n 16}` in two loads.
    ///
    /// # Panics
    ///
    /// Where `tile` names a dimension this space lacks, or does not divide its extent there.
    pub fn tiles(&self, tile: &Space) -> usize {
        tile.dimensions()
            .map(|dimension| {
                let (whole, part) = (self.extent(dimension), tile.extent(dimension));
                assert!(
                    whole.is_multiple_of(part),
                    "Space::tiles: {part} along {dimension:?} does not tile {whole}"
                );
                whole / part
            })
            .product()
    }

    /// The dimensions in this space but not in `output`: those contracted.
    pub(crate) fn difference(&self, output: &Space) -> SmallVec<[Dimension; Space::MAX_RANK]> {
        self.dimensions()
            .filter(|&dimension| !output.contains(dimension))
            .collect()
    }

    /// How many contracted values one step consumes off a `width`-wide line of this operand.
    pub(crate) fn contracted_per_step(&self, contracted: &[Dimension], width: usize) -> usize {
        let lined = self.dimension_at(self.rank() - 1);
        let whole_lines = match self.extent_raw(lined) {
            Extent::Static(extent) => extent.is_multiple_of(width),
            Extent::Dynamic => true,
        };
        let sums_lines = width > 1 && contracted.last() == Some(&lined) && whole_lines;
        if sums_lines { width } else { 1 }
    }

    /// The dimensions `operands` jointly contract against `output`.
    pub fn contracted(
        operands: &[&Space],
        output: &Space,
    ) -> SmallVec<[Dimension; Space::MAX_RANK]> {
        let merged = Space::merge(operands);
        // Read raw: asking a `Dynamic` extent's comptime size panics.
        let varies = |dimension: Dimension| merged.extent_raw(dimension) != Extent::Static(1);
        let shared =
            |dimension: Dimension| operands.iter().all(|operand| operand.contains(dimension));
        merged
            .difference(output)
            .into_iter()
            .filter(|&dimension| varies(dimension) || shared(dimension))
            .collect()
    }

    /// The `k` edge this operand contracts over against `output`.
    pub(crate) fn contracted_extent(&self, output: &Space) -> usize {
        self.difference(output)
            .iter()
            .map(|&dimension| self.extent(dimension))
            .product()
    }

    /// Whether `lhs` and `rhs` enumerate their jointly contracted dimensions in the same order.
    pub(crate) fn contraction_agrees(lhs: &Space, rhs: &Space, output: &Space) -> bool {
        let joint = Space::contracted(&[lhs, rhs], output);
        let listed = |operand: &Space| -> SmallVec<[Dimension; Space::MAX_RANK]> {
            operand
                .difference(output)
                .into_iter()
                .filter(|dimension| joint.contains(dimension))
                .collect()
        };
        listed(lhs) == listed(rhs)
    }

    /// How many cells the space holds.
    pub fn cells(&self) -> usize {
        self.dimensions()
            .map(|dimension| self.extent(dimension))
            .product()
    }

    /// How many slices along `dimension` the space holds: one for every index of its other
    /// dimensions. The extent of `dimension` itself may be one a walk has yet to cut, which only
    /// the launch knows.
    pub(crate) fn slices_along(&self, dimension: Dimension) -> usize {
        self.dimensions()
            .filter(|&other| other != dimension)
            .map(|other| self.extent(other))
            .product()
    }

    /// The widest run along `dimension` that divides both `widest` and this space's extent there:
    /// the widest line a buffer read in boxes of this space can hold without a box starting
    /// mid-line.
    pub(crate) fn whole_run(&self, dimension: Dimension, widest: usize) -> usize {
        gcd(widest, self.extent(dimension))
    }
}

#[cube]
impl Space {
    /// Dimension `i`'s extent, a dynamic extent as its runtime size.
    pub(crate) fn runtime_extent_at(&self, #[comptime] i: usize) -> usize {
        match comptime!(self.shape.extent_raw(self.shape.dimension_at(i))) {
            Extent::Static(n) => comptime!(n).runtime(),
            Extent::Dynamic => *self.sizes.index(i),
        }
    }
}

#[cfg(test)]
mod contraction_tests {
    use crate::*;

    const M: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);
    const R: Dimension = Dimension(3);

    /// A box holds its tiles side by side, one serving every position along a dimension it does
    /// not name.
    #[test]
    fn a_box_counts_the_tiles_it_holds() {
        let region = Space::new(&[(K, 16), (N, 16)]);
        let tiles = [
            Space::new(&[(K, 8)]),
            Space::new(&[(K, 16), (N, 16)]),
            Space::new(&[(K, 16), (N, 2)]),
        ];
        assert_eq!(tiles.map(|tile| region.tiles(&tile)), [2, 1, 8]);
    }

    /// A box divides another whose extent it divides along each of its own dimensions, whatever
    /// the other's along the rest.
    #[test]
    fn a_box_divides_along_its_own_dimensions() {
        let tile = Space::new(&[(K, 8)]);
        assert!(tile.divides(&Space::new(&[(N, 3), (K, 16)])));
    }

    /// A box a whole number of tiles long counts them.
    #[test]
    fn a_box_whole_tiles_long_counts_them() {
        let window = Space::new(&[(K, 64)]);
        assert_eq!(window.count_along(K, &Space::new(&[(K, 16)])), Some(4));
    }

    /// A box that cuts a tile short counts none.
    #[test]
    fn a_box_that_cuts_a_tile_short_counts_none() {
        let step = Space::new(&[(K, 16)]);
        assert_eq!(step.count_along(K, &Space::new(&[(K, 32)])), None);
    }

    /// A dimension the other box does not name is one wide, which only a box one wide divides.
    #[test]
    fn a_box_divides_no_box_lacking_a_dimension_it_reaches() {
        let tile = Space::new(&[(N, 2), (K, 8)]);
        assert!(!tile.divides(&Space::new(&[(K, 16)])));
    }

    /// A matmul's `k` is its one contracted dimension.
    #[test]
    fn a_matmul_contracts_its_one_dimension() {
        let lhs = Space::new(&[(M, 8), (K, 4)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 4);
    }

    /// A convolution's `k` is taps times channels.
    #[test]
    fn a_convolution_contracts_taps_times_channels() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 12);
    }

    /// Contracting nothing is `k = 1`.
    #[test]
    fn contracting_nothing_is_a_unit_depth() {
        let lhs = Space::new(&[(M, 8), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 1);
    }

    /// Same contraction order on both operands.
    #[test]
    fn operands_listing_one_contraction_order_agree() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = Space::new(&[(R, 3), (K, 4), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert!(Space::contraction_agrees(&lhs, &rhs, &out));
    }

    /// A permuted contraction order disagrees.
    #[test]
    fn a_permuted_contraction_order_disagrees() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = Space::new(&[(K, 4), (R, 3), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert!(!Space::contraction_agrees(&lhs, &rhs, &out));
    }
}
