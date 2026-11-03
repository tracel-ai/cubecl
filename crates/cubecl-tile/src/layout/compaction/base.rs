//! The smallest physical box a gathered operand's sub-tile can be staged in, and its mapping.

use cubecl::zspace::SmallVec;

use crate::algebra::gcd;
use crate::{Axis, PhysicalAxisMap, Projection, Scale, Space};

/// The compacted stage of a [`Projection`]: per physical axis, its cell count and source step,
/// plus the projection addressing it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Compaction {
    steps: SmallVec<[usize; Space::MAX_RANK]>,
    extents: SmallVec<[usize; Space::MAX_RANK]>,
    projection: Projection,
}

impl Compaction {
    /// Compact `projection`'s window over the sub-tile whose extents `extent_of` gives.
    pub fn new(
        projection: &Projection,
        vector_size: usize,
        extent_of: impl Fn(Axis) -> usize,
    ) -> Compaction {
        let rank = projection.physical_rank();
        let mut steps = SmallVec::new();
        let mut extents = SmallVec::new();
        let mut physical = Vec::with_capacity(rank);

        for pa in 0..rank {
            let axis_map = projection.physical_axis(pa);
            // At offset 0 on the stage, a static divisor may reduce even under a Dynamic offset.
            let scaled: Vec<(Axis, Scale)> =
                axis_map.terms().iter().map(|t| (t.axis, t.scale)).collect();
            let stage_map = PhysicalAxisMap::scaled(&scaled).over(axis_map.divisor());
            let terms = stage_map.terms();

            if stage_map.is_rational() {
                // A rational axis has no single step: dense, sized at its widest bounds.
                let d = stage_map.divisor().bound();
                let field: usize = terms
                    .iter()
                    .filter(|t| extent_of(t.axis) > 1)
                    .map(|t| (extent_of(t.axis) - 1) * t.scale.bound())
                    .sum();
                extents.push(1 + field.div_ceil(d));
                steps.push(1);
                physical.push(stage_map);
            } else {
                // Only moving terms constrain the step; a moving Dynamic coefficient forces step 1.
                let step = terms
                    .iter()
                    .filter(|t| extent_of(t.axis) > 1)
                    .try_fold(0, |g, t| match t.scale {
                        Scale::Static(s) => Some(gcd(g, s)),
                        Scale::Dynamic { .. } => None,
                    })
                    .unwrap_or(1);
                let step = step.max(1);
                // Non-moving static terms pin to 1; Dynamic ones keep their coefficient-array slot.
                let scaled: Vec<(Axis, Scale)> = terms
                    .iter()
                    .map(|t| {
                        let scale = match t.scale {
                            Scale::Static(s) if extent_of(t.axis) > 1 => Scale::Static(s / step),
                            Scale::Static(_) => Scale::Static(1),
                            dynamic => dynamic,
                        };
                        (t.axis, scale)
                    })
                    .collect();
                extents.push(
                    1 + scaled
                        .iter()
                        .map(|&(axis, scale)| (extent_of(axis) - 1) * scale.bound())
                        .sum::<usize>(),
                );
                steps.push(step);
                physical.push(PhysicalAxisMap::scaled(&scaled));
            }
        }

        let projection = Projection::new(projection.logical_axes(), &physical);
        // Compacted at expansion as well as on the host, where no caller takes a refusal.
        if let Err(refusal) = projection.validate(vector_size) {
            panic!("{refusal}");
        }

        Compaction {
            steps,
            extents,
            projection,
        }
    }

    /// How the stage's own logical axes address its cells.
    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// The step per physical axis: stage coordinate times step is the source coordinate.
    pub fn steps(&self) -> &[usize] {
        &self.steps
    }

    /// The cell count per physical axis, in elements.
    pub fn extents(&self) -> &[usize] {
        &self.extents
    }

    /// Whether the window has no holes, so a fill reads the source box straight through.
    pub(crate) fn is_dense(&self) -> bool {
        self.steps.iter().all(|&g| g == 1)
    }

    /// [`extents`](Compaction::extents) with the innermost in `vector_size`-wide lines.
    pub(crate) fn line_extents(&self, vector_size: usize) -> Vec<usize> {
        let last = self.extents.len() - 1;
        assert!(
            self.steps[last] == 1,
            "Compaction: the innermost physical axis is addressed in {vector_size}-wide lines, so \
             it must be an ungathered axis (extent {}, step {})",
            self.extents[last],
            self.steps[last]
        );
        let mut lines: Vec<usize> = self.extents.to_vec();
        lines[last] = lines[last].div_ceil(vector_size);
        lines
    }

    /// How many cells the stage allocates, in lines.
    pub fn cells(&self, vector_size: usize) -> usize {
        self.line_extents(vector_size).iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Divisor, Offset};

    const OH: Axis = Axis(0);
    const RH: Axis = Axis(1);
    const CI: Axis = Axis(2);

    /// `Ih <- Oh*stride + Rh*dilation`, `Ci` passed through.
    fn conv(stride: usize, dilation: usize) -> Projection {
        Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]),
                PhysicalAxisMap::of(CI),
            ],
        )
    }

    fn extents(oh: usize, rh: usize, ci: usize) -> impl Fn(Axis) -> usize {
        move |a| match a {
            OH => oh,
            RH => rh,
            _ => ci,
        }
    }

    fn extents2(oh: usize, rh: usize) -> impl Fn(Axis) -> usize {
        move |a| match a {
            OH => oh,
            RH => rh,
            _ => unreachable!("axis not spanned by this projection"),
        }
    }

    #[test]
    fn direct_compacts_to_itself() {
        let p = Projection::direct(&[OH, CI]);
        let c = Compaction::new(&p, 4, extents(8, 1, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[8, 16]);
        assert_eq!(c.projection(), &p);
    }

    #[test]
    fn a_unit_stride_window_is_the_receptive_field() {
        let c = Compaction::new(&conv(1, 1), 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[10, 16]);
        assert_eq!(c.cells(4), 10 * 4);
    }

    #[test]
    fn a_strided_window_with_adjacent_taps_stays_dense() {
        let c = Compaction::new(&conv(2, 1), 4, extents(8, 3, 16));
        assert!(c.is_dense());
        // 1 + 7*2 + 2*1
        assert_eq!(c.extents(), &[17, 16]);
    }

    #[test]
    fn a_single_tap_at_stride_two_halves_the_stage() {
        for dilation in [1, 3] {
            let c = Compaction::new(&conv(2, dilation), 4, extents(8, 1, 16));
            assert!(!c.is_dense());
            assert_eq!(c.steps(), &[2, 1]);
            assert_eq!(c.extents(), &[8, 16]);
            assert_eq!(c.projection().scale(0, OH), 1);
            assert_eq!(c.projection().scale(0, RH), 1);
        }
    }

    #[test]
    fn a_shared_factor_quotients_the_window() {
        let c = Compaction::new(&conv(2, 2), 4, extents(8, 3, 16));
        assert_eq!(c.steps(), &[2, 1]);
        // Bounding box 1 + 7*2 + 2*2 = 19, on the even lattice: 1 + 7 + 2 = 10.
        assert_eq!(c.extents(), &[10, 16]);
        assert_eq!(c.projection().scale(0, OH), 1);
        assert_eq!(c.projection().scale(0, RH), 1);
    }

    /// `{0, 1}` at stride 3 reaches `{0, 1, 3, 4, 6, 7}`: gcd 1, holes kept as padding.
    #[test]
    fn unreachable_offsets_inside_the_box_stay_as_padding() {
        let c = Compaction::new(&conv(3, 1), 4, extents(3, 2, 16));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[8, 16]);
    }

    #[test]
    fn a_ragged_innermost_extent_is_padded() {
        let lines = Compaction::new(&conv(1, 1), 4, extents(8, 3, 6)).line_extents(4);
        assert_eq!(lines, &[10, 2]);
    }

    #[test]
    fn a_scalar_gather_may_compact_its_only_axis() {
        let p = Projection::new(&[OH, RH], &[PhysicalAxisMap::affine(&[(OH, 2), (RH, 1)])]);
        let c = Compaction::new(&p, 1, extents2(8, 3));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[17]);
    }

    #[test]
    fn padded_projection_compacts_identically() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 1)]).shifted(-2),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::new(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[17, 16]);
        assert_eq!(c.projection().offset(0), Offset::Static(0));
    }

    #[test]
    fn dynamic_offset_projection_compacts_identically() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 1)]).shifted(Offset::Dynamic),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::new(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[17, 16]);
        assert_eq!(c.projection().offset(0), Offset::Static(0));
    }

    #[test]
    fn a_dynamic_coefficient_compacts_dense_against_its_max() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Dynamic { max: 2 }), (RH, Scale::Static(1))]),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::new(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        // 1 + 7*2 + 2*1, the box `conv(2, 1)` fills.
        assert_eq!(c.extents(), &[17, 16]);
        assert_eq!(
            Compaction::new(&conv(2, 1), 4, extents(8, 3, 16)).extents(),
            c.extents()
        );
        assert!(c.projection().physical_axis(0).has_dynamic_scale());
        assert_eq!(c.projection().dynamic_scale_index(0, 0), Some(0));
    }

    #[test]
    fn a_moving_dynamic_coefficient_gives_up_the_lattice() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Static(2)), (RH, Scale::Dynamic { max: 2 })]),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::new(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        // 1 + 7*2 + 2*2, the bounding box `conv(2, 2)` quotients but this one cannot.
        assert_eq!(c.extents(), &[19, 16]);
    }

    #[test]
    fn a_non_moving_dynamic_coefficient_keeps_its_array_slot() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Static(2)), (RH, Scale::Dynamic { max: 4 })]),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::new(&p, 4, extents(8, 1, 16));
        assert_eq!(c.steps(), &[2, 1]);
        assert_eq!(c.extents(), &[8, 16]);
        assert_eq!(p.dynamic_scale_index(0, 1), Some(0));
        assert_eq!(c.projection().dynamic_scale_index(0, 1), Some(0));
    }

    #[test]
    fn rational_axis_compacts_dense_with_conservative_extent() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(3),
                PhysicalAxisMap::of(CI),
            ],
        );
        // field = 7*2 + 2*3 = 20, extent = 1 + ⌈20 / 3⌉ = 8
        let c = Compaction::new(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[8, 16]);
        assert!(c.projection().physical_axis(0).is_rational());
        assert_eq!(c.projection().divisor(0), Divisor::Static(3));
    }

    #[test]
    fn a_dynamic_divisor_compacts_against_its_min() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(Divisor::Dynamic { min: 3 }),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::new(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[8, 16]);
        assert_eq!(c.projection().divisor(0), Divisor::Dynamic { min: 3 });
        assert_eq!(c.projection().dynamic_divisor_index(0), Some(0));
    }

    #[test]
    fn a_dynamic_divisor_box_holds_every_divisor_above_its_min() {
        let bounded = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(Divisor::Dynamic { min: 2 }),
                PhysicalAxisMap::of(CI),
            ],
        );
        let box_extent = Compaction::new(&bounded, 4, extents(8, 3, 16)).extents()[0];
        for d in 2..24 {
            let exact = Projection::new(
                &[OH, RH, CI],
                &[
                    PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(d),
                    PhysicalAxisMap::of(CI),
                ],
            );
            if exact.physical_axis(0).is_rational() {
                assert!(Compaction::new(&exact, 4, extents(8, 3, 16)).extents()[0] <= box_extent);
            }
        }
    }

    #[test]
    fn dynamic_offset_with_cancelling_divisor_compacts_as_integers() {
        let p_dynamic_offset = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Static(16)), (RH, Scale::Static(8))])
                    .shifted(Offset::Dynamic)
                    .over(4),
                PhysicalAxisMap::of(CI),
            ],
        );
        let p_integer = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 4), (RH, 2)]),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c_dynamic = Compaction::new(&p_dynamic_offset, 4, extents(8, 3, 16));
        let c_integer = Compaction::new(&p_integer, 4, extents(8, 3, 16));

        assert_eq!(c_dynamic.steps(), c_integer.steps());
        assert_eq!(c_dynamic.extents(), c_integer.extents());
        assert_eq!(c_dynamic.projection(), c_integer.projection());
        assert_eq!(c_dynamic.steps(), &[2, 1]);
    }
}
