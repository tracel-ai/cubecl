//! The labelled reading of a buffer the [`Arg`](super::Arg) builder derives ([`Labels`]).

use cubecl::zspace::{SmallVec, Tiling};

use crate::{Dimension, Geometry, Launcher, PhysicalAxisMap, Projection, Refusal, Space};

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
    ) -> Result<Self, Refusal> {
        let rank = geometry.rank();
        let block = geometry.labels(dimensions);
        if rank < block.len() {
            return Err(Refusal::RankBelowLabels {
                rank,
                block: block.len(),
            });
        }
        let batch_axes = rank - block.len();
        if batch_axes > batches.len() {
            return Err(Refusal::UnlabelledBatchAxes {
                axes: batch_axes,
                dimensions: batches.len(),
            });
        }
        let addressed: Vec<Dimension> = batches[batches.len() - batch_axes..]
            .iter()
            .chain(&block)
            .copied()
            .collect();

        let mut maps: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]> = SmallVec::new();
        let mut logical: SmallVec<[Dimension; Space::MAX_RANK]> = SmallVec::new();
        let mut axes: SmallVec<[(usize, usize); Space::MAX_RANK]> = SmallVec::new();
        for (&dimension, (extent, stride)) in addressed.iter().zip(geometry.axes()) {
            // A labelled dimension never drops out: the tile is shaped over it.
            if batches.contains(&dimension) && extent == 1 && !dimensions.contains(&dimension) {
                continue;
            }
            maps.push(PhysicalAxisMap::of(dimension));
            if !logical.contains(&dimension) {
                logical.push(dimension);
            }
            axes.push((extent, stride));
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
    ) -> Result<Self, Refusal> {
        if stored.is_tiled() {
            return Err(Refusal::GatherOfATiledBinding(stored));
        }
        if !(dimensions.is_empty() && batches.is_empty()) {
            return Err(Refusal::GatherWithLabels);
        }
        if projection.physical_rank() != geometry.rank() {
            return Err(Refusal::GatherRankMismatch {
                mapped: projection.physical_rank(),
                rank: geometry.rank(),
            });
        }
        let space = launch.partitioning().space();
        if let Some(&dimension) = projection
            .dimensions()
            .iter()
            .find(|&&dimension| !space.contains(dimension))
        {
            return Err(Refusal::GatherOfAnUnknownDimension(dimension));
        }
        Ok(Labels {
            geometry: geometry.clone(),
            addressed: projection.dimensions().to_vec(),
            projection,
        })
    }
}
