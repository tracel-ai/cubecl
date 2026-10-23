//! An operand's logical axes mapped onto its buffer's physical axes.

use crate::Addressed;
use cubecl::zspace::{SmallVec, Tiling};

use crate::{
    Axis, Composition, DimsBuilder, Divisor, Geometry, Offset, PhysicalAxisMap, Refusal, Scale,
    Space,
};

/// An operand's logical axes mapped onto its buffer's physical axes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Projection {
    /// One entry per physical axis, in buffer order.
    physical: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]>,
    /// The logical axes, in tile order; the last is the vectorized axis.
    axes: SmallVec<[Axis; Space::MAX_RANK]>,
}

impl Projection {
    /// One logical axis per physical axis at coefficient `1`, in buffer order.
    pub fn direct(axes: &[Axis]) -> Self {
        Projection {
            physical: axes.iter().map(|&a| PhysicalAxisMap::of(a)).collect(),
            axes: SmallVec::from_slice(axes),
        }
    }

    /// [`direct`](Projection::direct) over a space's own axes.
    pub(crate) fn direct_over(space: &crate::Space) -> Self {
        Projection::direct(&space.axes().collect::<Vec<_>>())
    }

    /// [`direct`](Projection::direct) storage-tiled: `labels` name each buffer dim by the axis it
    /// is a piece of, level-major.
    pub fn tiled(axes: &[Axis], labels: &[Axis]) -> Self {
        let physical: Vec<PhysicalAxisMap> =
            labels.iter().copied().map(PhysicalAxisMap::of).collect();
        Projection::new(axes, &physical)
    }

    /// The same operand with each axis's storage fragments merged back into one coordinate.
    pub(crate) fn untiled(&self) -> Projection {
        let carried = self.carried_groups();
        let physical: Vec<PhysicalAxisMap> = carried
            .iter()
            .map(|&pa| self.physical[pa].clone())
            .collect();
        Projection::new(&self.axes, &physical)
    }

    /// How many coordinates address this operand: its physical axes with tiled fragments folded.
    pub fn coordinate_rank(&self) -> usize {
        self.carried_groups().len()
    }

    /// The same buffer addressed by physical position: physical axis `p` relabeled `Axis(p)`.
    pub(crate) fn positional(&self) -> Projection {
        let carried = self.carried_groups();
        let axes: Vec<Axis> = (0..carried.len()).map(|p| Axis(p as u8)).collect();
        let physical: Vec<PhysicalAxisMap> = self
            .physical
            .iter()
            .enumerate()
            .map(|(pa, map)| {
                let at = match map.addressed() {
                    Addressed::Broadcast => carried.iter().position(|&q| q == pa),
                    Addressed::By(axis) => carried
                        .iter()
                        .position(|&q| self.physical[q].addressed() == Addressed::By(axis)),
                }
                .expect("collected above");
                PhysicalAxisMap::of(axes[at])
            })
            .collect();
        Projection::new(&axes, &physical)
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
                // Still a buffer axis, so it is its own group.
                Addressed::Broadcast => carried.push(pa),
                Addressed::By(axis) => {
                    if !carried
                        .iter()
                        .any(|&q| self.physical[q].addressed() == Addressed::By(axis))
                    {
                        carried.push(pa);
                    }
                }
            }
        }
        carried
    }

    /// How many fragments each logical axis is split across; panics past what a [`Tiling`]
    /// records.
    pub fn tiling(&self) -> Tiling {
        let fragments: Vec<usize> = self.axes.iter().map(|&axis| self.fragments(axis)).collect();
        Tiling::new(&fragments)
            .unwrap_or_else(|e| panic!("Projection::tiling: {fragments:?} fragments: {e:?}"))
    }

    /// Whether some axis is split across several physical fragments.
    pub(crate) fn is_tiled(&self) -> bool {
        self.axes.iter().any(|&axis| self.fragments(axis) > 1)
    }

    /// How many physical axes `axis` is split across. A broadcast axis is spread over none, which
    /// is not a tiling of it: one fragment, the same answer an untiled axis gives.
    fn fragments(&self, axis: Axis) -> usize {
        match self.addresses(axis) {
            false => 1,
            true => self.carriers(axis).len(),
        }
    }

    /// Where `axis`'s digit at physical axis `pa` sits: the positions of its finer fragments, and
    /// the one whose extent is this digit's radix (`None` for the outermost).
    pub(crate) fn digit(
        &self,
        pa: usize,
        axis: Axis,
    ) -> (SmallVec<[usize; Space::MAX_RANK]>, Option<usize>) {
        let carriers = self.carriers(axis);
        assert!(
            carriers.contains(&pa),
            "Projection::digit: physical axis {pa} does not carry {axis:?} (carried by {carriers:?})"
        );
        let finer = carriers.iter().copied().filter(|&q| q > pa).collect();
        (finer, (carriers[0] != pa).then_some(pa))
    }

    /// Whether this operand addresses `axis`; a spanned but unaddressed axis is a broadcast.
    pub(crate) fn addresses(&self, axis: Axis) -> bool {
        self.physical.iter().any(|m| m.addresses(axis))
    }

    /// The physical axes carrying `axis`, in buffer order. Panics if there are none.
    pub(crate) fn carriers(&self, axis: Axis) -> SmallVec<[usize; Space::MAX_RANK]> {
        let carriers: SmallVec<[usize; Space::MAX_RANK]> = (0..self.physical.len())
            .filter(|&q| self.physical[q].terms().iter().any(|t| t.axis == axis))
            .collect();
        assert!(
            !carriers.is_empty(),
            "Projection::carriers: {axis:?} addresses no physical axis of this operand"
        );
        carriers
    }

    /// This operand's scales' projection, one scale per `block`.
    /// `block` must partition its dim ([`disjoint`](PhysicalAxisMap::disjoint)); panics otherwise.
    pub fn scales_per(&self, block: Axis) -> Projection {
        let pa = self
            .physical
            .iter()
            .position(|map| map.terms().iter().any(|term| term.axis == block))
            .unwrap_or_else(|| {
                panic!("Projection::scales_per: {block:?} addresses no dim of this operand")
            });
        let map = &self.physical[pa];
        let terms = map.terms();
        let at = terms.iter().position(|term| term.axis == block).unwrap();
        assert!(
            map.composition() == Composition::Disjoint && at + 1 < terms.len(),
            "Projection::scales_per: {block:?} must partition its dim with a digit inside it \
             (PhysicalAxisMap::disjoint(&[({block:?}, block), (inside, 1)]))"
        );
        let radix = terms[at].scale.get();
        let kept: Vec<(Axis, usize)> = terms[..=at]
            .iter()
            .map(|term| (term.axis, term.scale.get() / radix))
            .collect();
        let omitted: Vec<Axis> = terms[at + 1..].iter().map(|term| term.axis).collect();
        let mut physical = self.physical.clone();
        physical[pa] = match kept.as_slice() {
            [(axis, 1)] => PhysicalAxisMap::of(*axis),
            _ => PhysicalAxisMap::disjoint(&kept),
        };
        Projection {
            physical,
            axes: self
                .axes
                .iter()
                .copied()
                .filter(|axis| !omitted.contains(axis))
                .collect(),
        }
    }

    /// `axes` in the tile's logical order, `physical` one per physical axis in buffer order.
    pub fn new(axes: &[Axis], physical: &[PhysicalAxisMap]) -> Self {
        Projection::of_maps(axes, physical.iter().cloned().collect())
    }

    /// [`new`](Projection::new) over maps the caller built for it, taken as they are.
    pub(crate) fn of_maps(
        axes: &[Axis],
        physical: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]>,
    ) -> Self {
        Projection {
            physical,
            axes: SmallVec::from_slice(axes),
        }
    }

    /// Whether this is the [`direct`](Projection::direct) mapping.
    pub(crate) fn is_direct(&self) -> bool {
        self.physical.len() == self.axes.len()
            && self
                .physical
                .iter()
                .zip(self.axes.iter())
                .all(|(map, &axis)| map.is_identity(axis))
    }

    /// The logical axes carried by the trailing physical axes, up to the innermost broadcast.
    pub(crate) fn dense_labels(&self) -> Vec<Axis> {
        let mut labels: Vec<Axis> = self
            .physical
            .iter()
            .rev()
            .map_while(|map| match map.addressed() {
                Addressed::By(axis) => Some(axis),
                Addressed::Broadcast => None,
            })
            .collect();
        labels.reverse();
        labels
    }

    pub fn physical_rank(&self) -> usize {
        self.physical.len()
    }

    pub(crate) fn logical_rank(&self) -> usize {
        self.axes.len()
    }

    pub(crate) fn logical_axes(&self) -> &[Axis] {
        &self.axes
    }

    /// `axis`'s index in the logical order, which is the order a coordinate comes in.
    pub fn position(&self, axis: Axis) -> usize {
        self.axes
            .iter()
            .position(|&a| a == axis)
            .expect("Projection::position: axis not spanned by this operand")
    }

    /// Every physical axis's map, in the projection's own order.
    fn axis_maps(&self) -> impl Iterator<Item = &PhysicalAxisMap> {
        self.physical.iter()
    }

    pub(crate) fn physical_axis(&self, pa: usize) -> &PhysicalAxisMap {
        &self.physical[pa]
    }

    /// `axis`'s coefficient along physical axis `pa`, `0` when it does not address it.
    pub fn scale(&self, pa: usize, axis: Axis) -> usize {
        self.physical[pa].scale(axis)
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

    /// Whether any physical axis has a negative or dynamic offset.
    pub(crate) fn may_underflow(&self) -> bool {
        self.physical.iter().any(|m| match m.offset() {
            Offset::Static(o) => o < 0,
            Offset::Dynamic => true,
        })
    }

    /// How many elements of physical axis `pa` a region covers: `1 + Σ (extent - 1) * scale`.
    /// Rational and dynamic axes report the widest span their bounds admit.
    pub(crate) fn span(&self, pa: usize, extent_of: impl Fn(Axis) -> usize) -> usize {
        let map = &self.physical[pa];
        let field: usize = map
            .terms()
            .iter()
            .map(|t| (extent_of(t.axis) - 1) * t.scale.bound())
            .sum();
        1 + field.div_ceil(map.divisor().bound())
    }

    /// `Disjoint` if every physical axis is partitioned by its axes, else `Overlapping`.
    pub(crate) fn composition(&self) -> Composition {
        match self
            .physical
            .iter()
            .all(|m| m.composition() == Composition::Disjoint && m.divisor().is_unit())
        {
            true => Composition::Disjoint,
            false => Composition::Overlapping,
        }
    }

    /// Panics if a disjoint axis's coefficients are not the products of its finer extents.
    pub(crate) fn validate_composition(&self, extent_of: impl Fn(Axis) -> usize) {
        for (pa, map) in self.physical.iter().enumerate() {
            let radices = map.claimed_radices();
            if radices.is_empty() {
                continue;
            }
            assert!(
                map.offset() == Offset::Static(0),
                "Projection: physical axis {pa} claims to be partitioned by its axes, but carries \
                 the offset {:?}; a partition starts at 0",
                map.offset()
            );
            // Only extents below the coarsest are read, keeping a `Dynamic` top axis out.
            let mut expected = 1;
            for (i, (axis, coefficient)) in radices.iter().enumerate().rev() {
                assert!(
                    *coefficient == expected,
                    "Projection: physical axis {pa} claims its axes partition it, so {axis:?} \
                     must step by {expected} (the extents below it); it steps by {coefficient}"
                );
                if i == 0 {
                    break;
                }
                expected *= extent_of(*axis);
            }
        }
    }

    pub(crate) fn is_invertible(&self) -> bool {
        self.physical.iter().all(|m| {
            m.offset() == Offset::Static(0)
                && m.divisor().is_unit()
                && matches!(m.terms(), [t] if matches!(t.scale, Scale::Static(1)))
        })
    }

    /// Why a gathered projection cannot be served at `vector_size`, if it cannot.
    pub(crate) fn validate(&self, vector_size: usize) -> Result<(), Refusal> {
        if self.is_invertible() {
            return Ok(());
        }
        if self.physical.is_empty() || self.axes.is_empty() {
            return Err(Refusal::NoCoordinate);
        }
        // The innermost physical axis is addressed in lines, so it must step by one element.
        if vector_size > 1 {
            let innermost = self.axes[self.axes.len() - 1];
            let last = &self.physical[self.physical.len() - 1];
            let steps_by_lines = match last.composition() {
                Composition::Disjoint => last.terms().last().map(|t| t.axis) == Some(innermost),
                Composition::Overlapping => last.is_identity(innermost),
            };
            if !steps_by_lines {
                return Err(Refusal::GatherInnermostNotInLines);
            }
        }
        for &axis in self.axes.iter() {
            // A gathered operand must be untiled: each addressed axis maps to one physical axis.
            let count = self.physical.iter().filter(|m| m.addresses(axis)).count();
            if count > 1 {
                return Err(Refusal::GatherAxisAddressedTwice(axis));
            }
        }
        Ok(())
    }

    /// Start writing a projection one buffer dim at a time, in buffer order ([`DimsBuilder`]).
    pub fn dims() -> DimsBuilder {
        DimsBuilder::new()
    }
}

// Queries against a buffer's runtime `Geometry`.
impl Projection {
    /// The axes carried by the buffer's unit-strided dim, or none where no dim strides by one.
    pub fn contiguous(&self, geometry: &Geometry) -> SmallVec<[Axis; Space::MAX_RANK]> {
        match self.unit_dim(geometry) {
            None => SmallVec::new(),
            Some(dim) => self
                .logical_axes()
                .iter()
                .copied()
                .filter(|&axis| self.physical_axis(dim).addresses(axis))
                .collect(),
        }
    }

    /// Whether the buffer's dims can be addressed without two positions landing on one cell.
    pub fn is_addressable(&self, geometry: &Geometry) -> bool {
        if self.composition() == Composition::Overlapping {
            return true;
        }
        by_stride(geometry).windows(2).all(|pair| {
            let (_, (_, stride)) = pair[0];
            let (_, (finer_extent, finer_stride)) = pair[1];
            stride >= finer_extent * finer_stride
        })
    }

    /// The buffer's contiguous dim — the innermost unit-strided one.
    fn unit_dim(&self, geometry: &Geometry) -> Option<usize> {
        by_stride(geometry)
            .into_iter()
            .filter(|&(_, (_, stride))| stride == 1)
            .map(|(dim, _)| dim)
            .next_back()
    }
}

// Indices into the runtime coefficient (unsigned) and offset (signed) arrays: physical axis
// major, term order within, each axis's divisor last.
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
        self.axis_maps().filter(|m| m.offset().is_dynamic()).count()
    }

    /// Whether any coefficient is only known at runtime.
    pub(crate) fn has_dynamic_scales(&self) -> bool {
        self.axis_maps().any(|m| m.has_dynamic_scale())
    }
}

/// The dims with extent above one and non-zero stride, as `(index, (extent, stride))`.
fn addressing_dims(geometry: &Geometry) -> SmallVec<[(usize, (usize, usize)); Space::MAX_RANK]> {
    geometry
        .dims()
        .enumerate()
        .filter(|&(_, (extent, stride))| extent > 1 && stride > 0)
        .collect()
}

/// The addressing dims, coarsest stride first.
fn by_stride(geometry: &Geometry) -> SmallVec<[(usize, (usize, usize)); Space::MAX_RANK]> {
    let mut dims = addressing_dims(geometry);
    dims.sort_by_key(|&(_, (_, stride))| core::cmp::Reverse(stride));
    dims
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Block scales omit the in-block digit and count the split dim in blocks.
    #[test]
    fn scales_per_block_omit_the_digit_inside_it() {
        let values = Projection::new(
            &[M, KB, KI],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::disjoint(&[(KB, 32), (KI, 1)]),
            ],
        );
        let scales = values.scales_per(KB);
        assert_eq!(scales.logical_axes(), &[M, KB]);
        assert_eq!(scales.physical_rank(), 2);
        assert!(scales.physical[1].is_identity(KB));
    }

    /// Scales per a middle digit keep the coarser digit, rescaled to count blocks.
    #[test]
    fn scales_per_a_coarser_digit_keep_what_is_above_it() {
        let values = Projection::new(
            &[M, KO, KB, KI],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::disjoint(&[(KO, 64), (KB, 32), (KI, 1)]),
            ],
        );
        let per_block = values.scales_per(KB);
        assert_eq!(per_block.logical_axes(), &[M, KO, KB]);
        assert_eq!(per_block.scale(1, KO), 2);
        assert_eq!(per_block.scale(1, KB), 1);
        let per_outer = values.scales_per(KO);
        assert_eq!(per_outer.logical_axes(), &[M, KO]);
        assert!(per_outer.physical[1].is_identity(KO));
    }

    #[test]
    #[should_panic(expected = "must partition its dim with a digit inside it")]
    fn scales_per_an_unsplit_axis_are_refused() {
        Projection::direct(&[M, KB]).scales_per(KB);
    }

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);
    const R: Axis = Axis(2);

    const M: Axis = Axis(3);
    const KB: Axis = Axis(4);
    const KI: Axis = Axis(5);
    const KO: Axis = Axis(6);

    #[test]
    fn direct_is_direct() {
        let p = Projection::direct(&[A, B]);
        assert!(p.is_direct());
        assert_eq!(p.physical_rank(), 2);
        assert_eq!(p.logical_axes(), &[A, B]);
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
        assert_eq!(refused, Err(Refusal::GatherAxisAddressedTwice(B)));
    }

    #[test]
    fn innermost_must_be_identity() {
        let refused = Projection::new(
            &[A, R],
            &[PhysicalAxisMap::of(A), PhysicalAxisMap::affine(&[(R, 2)])],
        )
        .validate(4);
        assert_eq!(refused, Err(Refusal::GatherInnermostNotInLines));
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
                PhysicalAxisMap::of(B),
            ],
        )
        .validate(1);
        assert_eq!(refused, Err(Refusal::GatherAxisAddressedTwice(B)));
    }

    #[test]
    fn a_gather_rejects_tiled_storage() {
        let refused = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
                PhysicalAxisMap::of(B),
            ],
        )
        .validate(4);
        assert_eq!(refused, Err(Refusal::GatherAxisAddressedTwice(B)));
    }

    #[test]
    fn an_axis_addressing_nothing_is_a_broadcast() {
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
        let p = Projection::tiled(&[A, B], &[A, B, A, B]);
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

        const BATCH: Axis = Axis(3);
        let spec = TileSpec::new(Projection::new(
            &[BATCH, A, B],
            &[BATCH, A, B, A, B].map(PhysicalAxisMap::of),
        ));

        assert_eq!(spec.projection.physical_rank(), 5);
        let [p0, p1, p2] = [Axis(0), Axis(1), Axis(2)];
        assert_eq!(
            spec.projection.positional(),
            Projection::tiled(&[p0, p1, p2], &[p0, p1, p2, p1, p2])
        );
        assert_eq!(
            Projection::tiled(&[BATCH, A, B], &[BATCH, A, B, A, B]),
            spec.projection
        );
        assert_eq!(spec.projection.tiling(), Tiling::new(&[1, 2, 2]).unwrap());
    }

    #[test]
    fn positional_of_a_plain_operand_is_the_identity() {
        assert_eq!(
            Projection::direct(&[B, A]).positional(),
            Projection::direct(&[Axis(0), Axis(1)])
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
        assert_eq!(p.positional(), Projection::direct(&[Axis(0), Axis(1)]));
    }

    #[test]
    fn a_ragged_tiling_orders_level_major() {
        let p = Projection::tiled(&[A, B], &[A, B, A, B, A]);
        assert_eq!(p.physical_rank(), 5);
        // `[A0, B0, A1, B1, A2]`.
        assert_eq!(p.carriers(A).as_slice(), &[0, 2, 4]);
        assert_eq!(p.carriers(B).as_slice(), &[1, 3]);
        assert_eq!(p.digit(0, A), (SmallVec::from_slice(&[2, 4]), None));
        assert_eq!(p.digit(2, A), (SmallVec::from_slice(&[4]), Some(2)));
        assert_eq!(p.digit(4, A), (SmallVec::new(), Some(4)));
        assert_eq!(p.digit(1, B), (SmallVec::from_slice(&[3]), None));
        assert!(p.is_invertible());
        p.validate(4).unwrap();
    }

    /// `tiling` reads back the piece counts of the dims `tiled` was given.
    #[test]
    fn tiling_reads_back_the_counts_tiled_was_built_from() {
        for fragments in [[1, 1, 1], [3, 3, 3], [1, 2, 2], [3, 1, 2]] {
            let axes: Vec<Axis> = (0..3).map(|p| Axis(p as u8)).collect();
            let labels = crate::StoragePartitioning::level_major(&axes, &fragments);
            let p = Projection::tiled(&axes, &labels);
            assert_eq!(p.tiling(), Tiling::new(&fragments).unwrap());
            assert_eq!(p.physical_rank(), fragments.iter().sum::<usize>());
            assert_eq!(p.is_tiled(), fragments.iter().any(|&n| n > 1));
        }
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

    #[test]
    fn may_underflow_tracks_negative_offsets_only() {
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
    fn rational_axis_conservative_span() {
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
        let extents = |axis| match axis {
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
        let extents = |axis| match axis {
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

    const K: Axis = Axis(0);
    const N: Axis = Axis(1);
    const B: Axis = Axis(2);
    const H: Axis = Axis(3);
    const S: Axis = Axis(4);
    const D: Axis = Axis(5);
    const M: Axis = Axis(6);
    const C: Axis = Axis(7);

    fn geometry(dims: &[(usize, usize)]) -> Geometry {
        Geometry::new(dims)
    }

    /// A `[k, n]` view over an `[n, k]` buffer: `k` strides by one, `n` by the row.
    #[test]
    fn a_col_weight_is_k_contiguous() {
        let p = Projection::direct(&[K, N]);
        let g = geometry(&[(4096, 1), (11008, 4096)]);

        assert_eq!(p.contiguous(&g).as_slice(), &[K]);
    }

    #[test]
    fn a_row_weight_is_n_contiguous() {
        let p = Projection::direct(&[K, N]);
        let g = geometry(&[(4096, 11008), (11008, 1)]);

        assert_eq!(p.contiguous(&g).as_slice(), &[N]);
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
        let p = Projection::dims()
            .dim(B)
            .dim(crate::layout::projection::dims_builder::stencil(&[(M, 2), (K, 1)]).pad(1))
            .dim(C)
            .build();
        let g = geometry(&[(8, 64 * 32), (64, 32), (32, 1)]);

        assert_eq!(p.composition(), Composition::Overlapping);
        assert!(p.is_addressable(&g));
        assert_eq!(p.contiguous(&g).as_slice(), &[C]);
    }
}
