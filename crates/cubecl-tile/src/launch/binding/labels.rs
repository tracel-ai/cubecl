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

        let mut kept: SmallVec<[Axis; Space::MAX_RANK]> = SmallVec::new();
        let mut dims: SmallVec<[(usize, usize); Space::MAX_RANK]> = SmallVec::new();
        for (&axis, (extent, stride)) in addressed.iter().zip(geometry.dims()) {
            // A labelled axis never drops out: the tile is shaped over it.
            if batches.contains(&axis) && extent == 1 && !axes.contains(&axis) {
                continue;
            }
            kept.push(axis);
            dims.push((extent, stride));
        }
        if !geometry.tiling().is_tiled() {
            let order = innermost_dim_order(&kept);
            kept = order.iter().map(|&dim| kept[dim]).collect();
            dims = order.iter().map(|&dim| dims[dim]).collect();
        }
        let maps: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]> =
            kept.iter().copied().map(PhysicalAxisMap::of).collect();
        let mut logical: SmallVec<[Axis; Space::MAX_RANK]> = SmallVec::new();
        for &axis in &kept {
            if !logical.contains(&axis) {
                logical.push(axis);
            }
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

/// The order a plain buffer's dims are bound in, `kept` naming the axis each carries: every axis's
/// dims just before its innermost one, each dim placed by its axis's innermost dim, in buffer order
/// within an axis. An axis spanning several dims (`[a, k, c]` labelled `[B, K, B]`, bound
/// `[k, a, c]` as `[K, B, B]`) then has its coordinate at its first dim in the order of the
/// innermost dims, so the innermost dim carries the last axis, the one a line runs along. Each dim
/// keeps its stride, so the buffer is read as it lies.
fn innermost_dim_order(kept: &[Axis]) -> SmallVec<[usize; Space::MAX_RANK]> {
    let innermost = |axis: Axis| {
        kept.iter()
            .rposition(|&other| other == axis)
            .expect("a kept dim's axis is kept")
    };
    let mut order: SmallVec<[usize; Space::MAX_RANK]> = (0..kept.len()).collect();
    order.sort_by_key(|&dim| (innermost(kept[dim]), dim));
    order
}
