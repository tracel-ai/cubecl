//! Where the [`Arg`](super::Arg) builder lands the bounds-check ([`Boundaries`]).

use cubecl::zspace::SmallVec;

use crate::BoundaryPolicy;
use crate::{Axis, Boundary, Projection, Refusal, Space};

/// Where the bounds-check lands: on the coordinate axes that can leave the buffer, only those.
pub(crate) struct Boundaries {
    /// One mode per coordinate axis; empty when nothing is checked.
    pub(crate) modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
}

impl Boundaries {
    /// The mode: the stated policy, else whether the operand overhangs, underflows or has an
    /// extent a table sets.
    fn mode(
        policy: Option<BoundaryPolicy>,
        projection: &Projection,
        addressed: &[Axis],
        concrete: &Space,
        overhangs: &[Axis],
        table_bounded: &[Axis],
    ) -> Option<Boundary> {
        match policy {
            Some(BoundaryPolicy::Unchecked) => None,
            Some(BoundaryPolicy::Every(boundary)) => Some(boundary),
            None => {
                let overhangs = addressed
                    .iter()
                    .filter(|&&axis| concrete.contains(axis))
                    .any(|axis| overhangs.contains(axis));
                (overhangs || projection.may_underflow() || !table_bounded.is_empty())
                    .then_some(Boundary::Zero)
            }
        }
    }

    /// `concrete` is the launch's real-extent space and `overhangs` the axes its tiles reach past.
    /// `table_bounded` are the axes a page table sets the extent of at launch
    /// ([`Paging`](crate::Paging)): never in bounds by construction, and their extent a whole
    /// number of pages, so of lines.
    pub(crate) fn new(
        policy: Option<BoundaryPolicy>,
        projection: &Projection,
        addressed: &[Axis],
        concrete: &Space,
        overhangs: &[Axis],
        table_bounded: &[Axis],
        width: usize,
    ) -> Result<Self, Refusal> {
        let coord_rank = projection.coordinate_rank();
        if coord_rank == 0 {
            return Err(Refusal::NoCoordinate);
        }
        // Most operands check nothing, and the untiled map below is the costly part.
        let Some(boundary) = Self::mode(
            policy,
            projection,
            addressed,
            concrete,
            overhangs,
            table_bounded,
        ) else {
            return Ok(Boundaries {
                modes: SmallVec::new(),
            });
        };
        let coords = projection.untiled();

        // Whether coordinate axis `pa` is always in bounds: it is one or more concrete axes laid
        // end to end (like `K = KB·16 + KI`), and the tiles reach past none of them and no table
        // bounds them.
        let settled = |pa: usize| {
            coords
                .physical_axis(pa)
                .flattened_axes(concrete)
                .is_some_and(|axes| {
                    axes.iter()
                        .all(|axis| !overhangs.contains(axis) && !table_bounded.contains(axis))
                })
        };
        // A table's extent is whole pages, each whole lines: a line never straddles its edge.
        let whole_lines = |pa: usize| {
            coords
                .physical_axis(pa)
                .identity_axis()
                .is_some_and(|axis| table_bounded.contains(&axis))
        };

        if width > 1 && !settled(coord_rank - 1) && !whole_lines(coord_rank - 1) {
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
