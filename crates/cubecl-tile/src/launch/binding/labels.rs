//! The labelled reading of a buffer the [`Arg`](super::Arg) builder derives ([`Labels`]).

use cubecl::zspace::{SmallVec, Tiling};

use crate::{Decline, Dimension, Geometry, Launcher, PhysicalAxisMap, Projection, Space};

/// The labelled reading of a buffer, as a [`Projection`] and the settled geometry.
pub(crate) struct Labels {
    /// The buffer less its dropped axes, its tiling restated over the axes it keeps: a tiling
    /// counts pieces from the leading axis, so one counted off a dropped axis claims a layout the
    /// buffer lacks.
    pub(crate) geometry: Geometry,
    pub(crate) projection: Projection,
    /// Every dimension the buffer carries, before broadcast axes dropped.
    pub(crate) addressed: Vec<Dimension>,
}

impl Labels {
    pub(crate) fn new(
        geometry: &Geometry,
        dimensions: &[Dimension],
        batches: &[Dimension],
    ) -> Result<Self, Decline> {
        let rank = geometry.rank();
        let block = geometry.labels(dimensions);
        if rank < block.len() {
            return Err(Decline::RankBelowLabels {
                rank,
                block: block.len(),
            });
        }
        let batch_axes = rank - block.len();
        if batch_axes > batches.len() {
            return Err(Decline::UnlabelledBatchAxes {
                axes: batch_axes,
                dimensions: batches.len(),
            });
        }
        let addressed: Vec<Dimension> = batches[batches.len() - batch_axes..]
            .iter()
            .chain(&block)
            .copied()
            .collect();

        let mut kept: SmallVec<[Dimension; Space::MAX_RANK]> = SmallVec::new();
        let mut axes: SmallVec<[(usize, usize); Space::MAX_RANK]> = SmallVec::new();
        for (&dimension, (extent, stride)) in addressed.iter().zip(geometry.axes()) {
            // A labelled dimension never drops out: the tile is shaped over it.
            if batches.contains(&dimension) && extent == 1 && !dimensions.contains(&dimension) {
                continue;
            }
            kept.push(dimension);
            axes.push((extent, stride));
        }
        if !geometry.tiling().is_tiled() {
            let order = innermost_axis_order(&kept);
            kept = order.iter().map(|&axis| kept[axis]).collect();
            axes = order.iter().map(|&axis| axes[axis]).collect();
        }
        let maps: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]> =
            kept.iter().copied().map(PhysicalAxisMap::of).collect();
        let mut logical: SmallVec<[Dimension; Space::MAX_RANK]> = SmallVec::new();
        for &dimension in &kept {
            if !logical.contains(&dimension) {
                logical.push(dimension);
            }
        }
        // The batch axes kept, one piece each, then each labelled dimension's pieces.
        let pieces = dimensions
            .iter()
            .map(|&dimension| block.iter().filter(|&&a| a == dimension).count());
        let fragments: Vec<usize> = core::iter::repeat_n(1, axes.len() - block.len())
            .chain(pieces)
            .collect();
        let tiling = match geometry.tiling().is_tiled() {
            true => Tiling::new(&fragments).expect("the binding's own tiling, less plain axes"),
            false => Tiling::UNTILED,
        };
        Ok(Labels {
            geometry: Geometry::new(&axes).with_tiling(tiling),
            projection: Projection::of_maps(&logical, maps),
            addressed,
        })
    }

    /// A stated gathered mapping, checked against the binding and the launch.
    pub(crate) fn stated(
        geometry: &Geometry,
        launch: &Launcher,
        projection: Projection,
        dimensions: &[Dimension],
        batches: &[Dimension],
        stored: Tiling,
    ) -> Result<Self, Decline> {
        if stored.is_tiled() {
            return Err(Decline::GatherOfATiledBinding(stored));
        }
        assert!(
            dimensions.is_empty() && batches.is_empty(),
            "Arg::gathered: the mapping is stated outright, so `dimensions` and `batches` have \
             nothing left to describe"
        );
        if projection.physical_rank() != geometry.rank() {
            return Err(Decline::GatherRankMismatch {
                mapped: projection.physical_rank(),
                rank: geometry.rank(),
            });
        }
        let space = launch.partitioning().space();
        let unknown = projection
            .dimensions()
            .iter()
            .find(|&&dimension| !space.contains(dimension));
        assert!(
            unknown.is_none(),
            "Arg::gathered: the mapping spans {unknown:?}, which the launched space does not have"
        );
        Ok(Labels {
            geometry: geometry.clone(),
            addressed: projection.dimensions().to_vec(),
            projection,
        })
    }
}

/// The order a plain buffer's axes are bound in, `kept` naming the dimension each carries: every
/// dimension's axes just before its innermost one, each axis placed by its dimension's innermost
/// axis, in buffer order within a dimension. A dimension spanning several axes (`[a, k, c]`
/// labelled `[B, K, B]`, bound `[k, a, c]` as `[K, B, B]`) then has its coordinate at its first
/// axis in the order of the innermost axes, so the innermost axis carries the last dimension, the
/// one a line runs along. Each axis keeps its stride, so the buffer is read as it lies.
fn innermost_axis_order(kept: &[Dimension]) -> SmallVec<[usize; Space::MAX_RANK]> {
    let innermost = |dimension: Dimension| {
        kept.iter()
            .rposition(|&other| other == dimension)
            .expect("a kept axis's dimension is kept")
    };
    let mut order: SmallVec<[usize; Space::MAX_RANK]> = (0..kept.len()).collect();
    order.sort_by_key(|&axis| (innermost(kept[axis]), axis));
    order
}
