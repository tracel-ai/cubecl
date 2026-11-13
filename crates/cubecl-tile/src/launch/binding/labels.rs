//! The labelled reading of a buffer the [`Arg`](super::Arg) builder derives ([`Labels`]).

use cubecl::zspace::{SmallVec, Tiling};

use crate::{Axis, Geometry, Launcher, PhysicalAxisMap, Projection, Refusal, Space};

/// The labelled reading of a buffer, as a [`Projection`] and the settled geometry.
pub(crate) struct Labels {
    /// The buffer less its dropped dims, its tiling restated over the dims it keeps: a tiling
    /// counts pieces from the leading dim, so one counted off a dropped dim claims a layout the
    /// buffer lacks.
    pub(crate) geometry: Geometry,
    pub(crate) projection: Projection,
    /// Every logical axis the buffer carries, before broadcast dims dropped.
    pub(crate) addressed: Vec<Axis>,
}

impl Labels {
    pub(crate) fn new(
        geometry: &Geometry,
        axes: &[Axis],
        batches: &[Axis],
    ) -> Result<Self, Refusal> {
        let rank = geometry.rank();
        let block = geometry.labels(axes);
        if rank < block.len() {
            return Err(Refusal::RankBelowLabels {
                rank,
                block: block.len(),
            });
        }
        let batch_dims = rank - block.len();
        if batch_dims > batches.len() {
            return Err(Refusal::UnlabelledBatchDims {
                dims: batch_dims,
                axes: batches.len(),
            });
        }
        let addressed: Vec<Axis> = batches[batches.len() - batch_dims..]
            .iter()
            .chain(&block)
            .copied()
            .collect();

        let mut maps: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]> = SmallVec::new();
        let mut logical: SmallVec<[Axis; Space::MAX_RANK]> = SmallVec::new();
        let mut dims: SmallVec<[(usize, usize); Space::MAX_RANK]> = SmallVec::new();
        for (&axis, (extent, stride)) in addressed.iter().zip(geometry.dims()) {
            // A labelled axis never drops out: the tile is shaped over it.
            if batches.contains(&axis) && extent == 1 && !axes.contains(&axis) {
                continue;
            }
            maps.push(PhysicalAxisMap::of(axis));
            if !logical.contains(&axis) {
                logical.push(axis);
            }
            dims.push((extent, stride));
        }
        // The batch dims kept, one piece each, then each labelled axis's pieces.
        let pieces = axes
            .iter()
            .map(|&axis| block.iter().filter(|&&a| a == axis).count());
        let fragments: Vec<usize> = core::iter::repeat_n(1, dims.len() - block.len())
            .chain(pieces)
            .collect();
        let tiling = match geometry.tiling().is_tiled() {
            true => Tiling::new(&fragments).expect("the binding's own tiling, less plain dims"),
            false => Tiling::UNTILED,
        };
        Ok(Labels {
            geometry: Geometry::new(&dims).with_tiling(tiling),
            projection: Projection::of_maps(&logical, maps),
            addressed,
        })
    }

    /// A stated gathered mapping, checked against the binding and the launch.
    pub(crate) fn stated(
        geometry: &Geometry,
        launch: &Launcher,
        projection: Projection,
        axes: &[Axis],
        batches: &[Axis],
        stored: Tiling,
    ) -> Result<Self, Refusal> {
        if stored.is_tiled() {
            return Err(Refusal::GatherOfATiledBinding(stored));
        }
        if !(axes.is_empty() && batches.is_empty()) {
            return Err(Refusal::GatherWithLabels);
        }
        if projection.physical_rank() != geometry.rank() {
            return Err(Refusal::GatherRankMismatch {
                mapped: projection.physical_rank(),
                rank: geometry.rank(),
            });
        }
        let space = launch.partitioning().space();
        if let Some(&axis) = projection
            .logical_axes()
            .iter()
            .find(|&&axis| !space.contains(axis))
        {
            return Err(Refusal::GatherOfAnUnknownAxis(axis));
        }
        Ok(Labels {
            geometry: geometry.clone(),
            addressed: projection.logical_axes().to_vec(),
            projection,
        })
    }
}
