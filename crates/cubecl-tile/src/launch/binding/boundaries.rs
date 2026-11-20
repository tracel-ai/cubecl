//! Where the [`Arg`](super::Arg) builder lands the bounds-check ([`Boundaries`]).

use cubecl::zspace::SmallVec;

use crate::BoundaryPolicy;
use crate::{Boundary, Dimension, Projection, Refusal, Space};

/// Where the bounds-check lands: on the coordinate dimensions that can leave the buffer, only
/// those.
pub(crate) struct Boundaries {
    /// One mode per coordinate dimension; empty when nothing is checked.
    pub(crate) modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
}

impl Boundaries {
    /// The mode: the stated policy, else, where the operand overhangs or underflows, the boundary
    /// [`AnyExtent`](BoundaryPolicy::AnyExtent) states or [`Boundary::Zero`].
    fn mode(
        policy: Option<BoundaryPolicy>,
        projection: &Projection,
        addressed: &[Dimension],
        concrete: &Space,
        overhangs: &[Dimension],
    ) -> Option<Boundary> {
        let boundary = match policy {
            Some(BoundaryPolicy::Unchecked) => return None,
            Some(BoundaryPolicy::Every(boundary)) => return Some(boundary),
            Some(BoundaryPolicy::AnyExtent(boundary)) => boundary,
            None => Boundary::Zero,
        };
        let overhangs = addressed
            .iter()
            .filter(|&&dimension| concrete.contains(dimension))
            .any(|dimension| overhangs.contains(dimension));
        (overhangs || projection.may_underflow()).then_some(boundary)
    }

    /// `concrete` is the launch's real-extent space and `overhangs` the dimensions its tiles reach
    /// past: this launch's, or under [`AnyExtent`](BoundaryPolicy::AnyExtent) those any extent
    /// could.
    pub(crate) fn new(
        policy: Option<BoundaryPolicy>,
        projection: &Projection,
        addressed: &[Dimension],
        concrete: &Space,
        overhangs: &[Dimension],
        width: usize,
    ) -> Result<Self, Refusal> {
        let coord_rank = projection.coordinate_rank();
        if coord_rank == 0 {
            return Err(Refusal::NoCoordinate);
        }
        // Most operands check nothing, and the untiled map below is the costly part.
        let Some(boundary) = Self::mode(policy, projection, addressed, concrete, overhangs) else {
            return Ok(Boundaries {
                modes: SmallVec::new(),
            });
        };
        let coords = projection.untiled();

        // Whether coordinate dimension `pa` is always in bounds: it is one or more concrete
        // dimensions laid end to end (like `K = KB·16 + KI`), and the tiles reach past none of
        // them.
        let settled = |pa: usize| {
            coords
                .physical_axis(pa)
                .flattened_dimensions(concrete)
                .is_some_and(|dimensions| {
                    dimensions
                        .iter()
                        .all(|dimension| !overhangs.contains(dimension))
                })
        };

        // Whether coordinate dimension `pa` is checked whole lines at a time without clipping one:
        // it is a single concrete dimension, the map never reaches below its start, and `width`
        // divides its extent, so every line lies wholly before the edge or wholly past it. Only a
        // zeroing mask is tested a line at a time, as a clamp moves each element; and only one this
        // launch's extent decides, since under `AnyExtent` the kernel serves extents the width need
        // not divide.
        let one_extent = !matches!(policy, Some(BoundaryPolicy::AnyExtent(_)));
        let whole_lines = |pa: usize| {
            one_extent
                && boundary == Boundary::Zero
                && !projection.may_underflow()
                && coords
                    .physical_axis(pa)
                    .flattened_dimensions(concrete)
                    .is_some_and(|dimensions| match dimensions.as_slice() {
                        [dimension] => concrete.extent(*dimension).is_multiple_of(width),
                        _ => false,
                    })
        };

        let innermost = coord_rank - 1;
        if width > 1 && !settled(innermost) && !whole_lines(innermost) {
            return Err(Refusal::UncheckableVectorEdge);
        }

        let modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]> = (0..coord_rank)
            .map(|pa| (!settled(pa)).then_some(boundary))
            .collect();
        // "Nothing is checked" has one form: the empty list.
        Ok(match modes.iter().any(Option::is_some) {
            true => Boundaries { modes },
            false => Boundaries {
                modes: SmallVec::new(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PhysicalAxisMap, Scale};

    const OH: Dimension = Dimension(0);
    const RH: Dimension = Dimension(1);
    const CI: Dimension = Dimension(2);

    #[test]
    fn a_backward_map_with_no_policy_zeroes_its_spatial_dimension_only() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Static(1)), (RH, Scale::Static(-1))])
                    .shifted(1),
                PhysicalAxisMap::of(CI),
            ],
        );
        let concrete = Space::new(&[(OH, 8), (RH, 3), (CI, 4)]);
        let boundaries = Boundaries::new(None, &p, &[OH, RH, CI], &concrete, &[], 1).unwrap();
        assert_eq!(boundaries.modes.as_slice(), &[Some(Boundary::Zero), None]);
    }
}
