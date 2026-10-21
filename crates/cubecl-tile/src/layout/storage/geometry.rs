//! An operand's extents and strides: [`Geometry`] on the host, [`RuntimeGeometry`] in the kernel.

use cubecl::prelude::*;

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{SmallVec, Tiling};

use crate::{Axis, StoragePartitioning, TileMisfit};

use crate::Coords;

/// One operand's physical `(extent, stride)` per dim, in scalars, and its storage tiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    shape: Vec<usize>,
    strides: Vec<usize>,
    tiling: Tiling,
}

impl Geometry {
    /// One `(extent, stride)` per physical dim, coarsest first, both in scalars.
    pub fn new(dims: &[(usize, usize)]) -> Self {
        Self {
            shape: dims.iter().map(|&(extent, _)| extent).collect(),
            strides: dims.iter().map(|&(_, stride)| stride).collect(),
            tiling: Tiling::UNTILED,
        }
    }

    /// A dense row-major buffer of `shape`, coarsest first: what an output with no tensor behind it
    /// is counted in.
    pub fn row_major(shape: &[usize]) -> Self {
        let mut strides = vec![1; shape.len()];
        for dim in (0..shape.len().saturating_sub(1)).rev() {
            strides[dim] = strides[dim + 1] * shape[dim + 1];
        }
        Self {
            shape: shape.to_vec(),
            strides,
            tiling: Tiling::UNTILED,
        }
    }

    /// This geometry with its dims stated as pieces of logical dims, as `tiling` says.
    pub(crate) fn with_tiling(mut self, tiling: Tiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// Which dims are pieces of one logical dim: untiled unless the binding said otherwise.
    pub fn tiling(&self) -> Tiling {
        self.tiling
    }

    /// Its trailing dims, coarsest first, each named by the axis it is a piece of: `axes` name the
    /// trailing logical dims in the tensor's order, each once per piece its tiling stores it as, in
    /// [`StoragePartitioning::labels`]' order. The dims ahead of them are batch dims, one each.
    ///
    /// # Panics
    ///
    /// When the tiling does not fit the rank, when the buffer stands for fewer logical dims than
    /// `axes` name, or when it tiles a batch dim, which is stored one dim.
    pub fn labels(&self, axes: &[Axis]) -> Vec<Axis> {
        if !self.tiling.is_tiled() {
            return axes.to_vec();
        }
        let logical_rank = self
            .tiling
            .logical_rank(self.rank())
            .unwrap_or_else(|e| panic!("Geometry::labels: {e:?}"));
        assert!(
            logical_rank >= axes.len(),
            "Geometry::labels: the buffer stands for {logical_rank} dims but {} axes name them",
            axes.len()
        );
        let fragments = self.tiling.fragments(logical_rank);
        let (batches, block) = fragments.split_at(logical_rank - axes.len());
        assert!(
            batches.iter().all(|&n| n == 1),
            "Geometry::labels: batch dims are stored one dim each, so {batches:?} pieces over \
             them cannot be named"
        );
        StoragePartitioning::level_major(axes, block)
    }

    /// The extents, coarsest first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides that step them, in scalars.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// How many physical dims the operand has.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// This geometry with its trailing `labels.len()` dims and labels reordered by stride, coarsest
    /// first; leading (batch) dims stay put.
    pub fn in_stride_order(&self, labels: &[Axis]) -> (Geometry, Vec<Axis>) {
        let batch_dims = self.rank() - labels.len();
        let mut trailing: Vec<(usize, (usize, usize))> =
            self.dims().enumerate().skip(batch_dims).collect();
        trailing.sort_by_key(|&(_, (_, stride))| core::cmp::Reverse(stride));
        let dims: Vec<(usize, usize)> = self
            .dims()
            .take(batch_dims)
            .chain(trailing.iter().map(|&(_, dim)| dim))
            .collect();
        let ordered = trailing
            .iter()
            .map(|&(dim, _)| labels[dim - batch_dims])
            .collect();
        (Geometry::new(&dims), ordered)
    }

    pub fn dims(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.shape.iter().copied().zip(self.strides.iter().copied())
    }

    /// Whether this buffer can be read a `tile` at a time, `tile` finest first over the axes
    /// `labels` name, right-aligned to the dims.
    pub fn serves(&self, tile: &[(Axis, usize)], labels: &[Axis]) -> Result<(), LineMisfit> {
        let values: usize = tile.iter().map(|&(_, count)| count).product();
        if values == 1 {
            return Ok(());
        }
        if self.rank() == 0 {
            return Err(LineMisfit::NoDims);
        }
        let storage = StoragePartitioning::new(self, labels).ok_or(LineMisfit::NotPartitioned)?;
        storage
            .holds(tile, &self.extents(labels))
            .map_err(LineMisfit::Tile)?;
        // The run, from the finest dim up: each dim it takes steps by what it has taken so far.
        let mut by_stride: SmallVec<[(usize, (usize, usize)); 8]> = self
            .dims()
            .enumerate()
            .filter(|&(_, (extent, stride))| extent > 1 && stride > 0)
            .collect();
        by_stride.sort_by_key(|&(dim, (_, stride))| (stride, core::cmp::Reverse(dim)));
        let mut inside: SmallVec<[usize; 8]> = SmallVec::new();
        let mut run = 1;
        for &(dim, (extent, stride)) in &by_stride {
            if run >= values {
                break;
            }
            if stride != run {
                return Err(match run {
                    1 => LineMisfit::InnermostStrided(stride),
                    _ => LineMisfit::Gap(stride),
                });
            }
            inside.push(dim);
            run *= extent;
        }
        match self
            .dims()
            .enumerate()
            .find(|&(dim, (_, stride))| !inside.contains(&dim) && !stride.is_multiple_of(values))
        {
            Some((_, (_, stride))) => Err(LineMisfit::StrideInsideLine(stride)),
            None => Ok(()),
        }
    }

    /// Each labelled axis's extent: the product of the dims it labels.
    pub(crate) fn extents(&self, labels: &[Axis]) -> Vec<(Axis, usize)> {
        let unlabelled = self.rank().saturating_sub(labels.len());
        let mut extents: Vec<(Axis, usize)> = Vec::new();
        for (dim, extent) in self.shape.iter().copied().enumerate().skip(unlabelled) {
            let axis = labels[dim - unlabelled];
            match extents.iter_mut().find(|(a, _)| *a == axis) {
                Some((_, product)) => *product *= extent,
                None => extents.push((axis, extent)),
            }
        }
        extents
    }
}

/// Why a [`Geometry`] cannot be read a tile at a time ([`Geometry::serves`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineMisfit {
    /// The finest dim's own stride, when it is not 1: consecutive values are not one run.
    InnermostStrided(usize),
    /// A dim inside the tile that does not step by what is finer than it: a gap in the run.
    Gap(usize),
    /// The storage partitioning does not hold the tile: see [`TileMisfit`].
    Tile(TileMisfit),
    /// A stride outside the tile that is not a whole number of tiles.
    StrideInsideLine(usize),
    /// The buffer's order is no partitioning: a stated piece coarser than the rest of an axis.
    NotPartitioned,
    /// No dims at all, so only a single value fits.
    NoDims,
}

impl Display for LineMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InnermostStrided(stride) => write!(
                f,
                "its finest dim steps by {stride} rather than 1, so consecutive values are not \
                 one run"
            ),
            Self::Gap(stride) => write!(
                f,
                "a dim inside the tile steps by {stride}, leaving a gap in its run"
            ),
            Self::Tile(why) => write!(f, "its storage does not hold the tile: {why}"),
            Self::StrideInsideLine(stride) => write!(
                f,
                "its stride {stride} is not a whole number of tiles, so a coarser step lands \
                 inside one"
            ),
            Self::NotPartitioned => write!(
                f,
                "a stated piece sits coarser than the rest of an axis, so its order is no \
                 partitioning"
            ),
            Self::NoDims => write!(f, "it has no dims, so only a single value fits"),
        }
    }
}

impl From<&TensorBinding> for Geometry {
    fn from(binding: &TensorBinding) -> Self {
        Self {
            shape: binding.shape.to_vec(),
            strides: binding.strides.to_vec(),
            tiling: binding.tiling,
        }
    }
}

/// [`Geometry`]'s kernel-side twin.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RuntimeGeometry {
    pub(crate) shape: Coords<u32>,
    pub(crate) strides: Coords<u32>,
}

#[cube]
impl RuntimeGeometry {
    /// An empty geometry, grown a dim at a time by `push`.
    // A cube constructor: `Default` has no kernel expansion.
    #[allow(clippy::new_without_default)]
    pub fn new() -> RuntimeGeometry {
        RuntimeGeometry {
            shape: Coords::<u32>::new(),
            strides: Coords::<u32>::new(),
        }
    }

    /// The geometry a launched tensor carries, over its first `rank` dims.
    pub fn of_tensor<E: CubePrimitive>(
        tensor: &Tensor<E>,
        #[comptime] rank: usize,
    ) -> RuntimeGeometry {
        let mut geometry = RuntimeGeometry::new();
        #[unroll]
        for i in 0..rank {
            geometry.push(tensor.shape(i) as u32, tensor.stride(i) as u32);
        }
        geometry
    }

    /// One physical dim, coarsest first: how far it runs, and the scalar stride that steps it.
    pub fn push(&mut self, extent: u32, stride: u32) {
        self.shape.push(extent);
        self.strides.push(stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageLevels;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    #[test]
    fn stride_order_reorders_the_dims_and_their_labels() {
        // `[n, k]` behind a `[k, n]` view: `k` strides by one.
        let (dims, labels) = Geometry::new(&[(4096, 1), (6144, 4096)]).in_stride_order(&[K, N]);
        assert_eq!(labels, vec![N, K]);
        assert_eq!(dims, Geometry::new(&[(6144, 4096), (4096, 1)]));

        // Already in order: nothing moves.
        let (dims, labels) = Geometry::new(&[(4096, 6144), (6144, 1)]).in_stride_order(&[K, N]);
        assert_eq!(labels, vec![K, N]);
        assert_eq!(dims, Geometry::new(&[(4096, 6144), (6144, 1)]));

        // A broadcast batch dim strides by zero and stays first all the same.
        let (dims, labels) =
            Geometry::new(&[(8, 0), (4096, 1), (6144, 4096)]).in_stride_order(&[K, N]);
        assert_eq!(labels, vec![N, K]);
        assert_eq!(dims, Geometry::new(&[(8, 0), (6144, 4096), (4096, 1)]));
    }

    /// On untiled buffers, `serves` agrees with the one-axis line rule ([`line_rule_serves`]).
    #[test]
    fn a_line_is_served_where_the_line_rule_served_it() {
        let geometries: Vec<(Geometry, Vec<Axis>)> = vec![
            (Geometry::new(&[(64, 128), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(64, 136), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(64, 130), (130, 1)]), vec![M, N]),
            (Geometry::new(&[(64, 1), (128, 64)]), vec![M, N]),
            (Geometry::new(&[(3, 0), (64, 128), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(1, 5), (64, 128), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(2, 4096), (32, 1)]), vec![K, N]),
            (
                Geometry::new(&[(4, 2048), (4, 512), (16, 32), (32, 1)]),
                vec![K, N, K, N],
            ),
            (Geometry::new(&[(6, 1)]), vec![N]),
        ];
        for (geometry, labels) in &geometries {
            let line = *labels.last().unwrap();
            for width in [1, 2, 3, 4, 8, 16, 32] {
                assert_eq!(
                    geometry.serves(&[(line, width)], labels).is_ok(),
                    line_rule_serves(geometry, width),
                    "{geometry:?} at {width}"
                );
            }
        }
    }

    #[test]
    fn a_tiled_buffer_serves_the_reads_it_was_stored_for() {
        let storage = StorageLevels::new(&[(N, 4)])
            .tile(&[(N, 8), (K, 32)])
            .grid(&[N, K]);
        let geometry = storage.physical(&[(K, 64), (N, 64)]).unwrap();
        let labels = geometry.labels(&[K, N]);
        assert_eq!(geometry.serves(&[(N, 4)], &labels), Ok(()));
        assert_eq!(geometry.serves(&[(N, 32), (K, 32)], &labels), Ok(()));
        assert!(geometry.serves(&[(N, 2)], &labels).is_err());
        assert!(geometry.serves(&[(N, 8)], &labels).is_err());
    }

    /// The one-axis line rule: a line of `width` is served where the innermost dim is contiguous
    /// and its extent and every outer stride are multiples of `width`.
    fn line_rule_serves(geometry: &Geometry, width: usize) -> bool {
        if width == 1 {
            return true;
        }
        let Some(last) = geometry.rank().checked_sub(1) else {
            return false;
        };
        geometry.strides()[last] == 1
            && geometry.shape()[last].is_multiple_of(width)
            && geometry.strides()[..last]
                .iter()
                .all(|s| s.is_multiple_of(width))
    }
}
