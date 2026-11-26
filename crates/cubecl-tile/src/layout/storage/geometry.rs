//! An operand's extents and strides: [`Geometry`] on the host, [`RuntimeGeometry`] in the kernel.

use cubecl::prelude::*;

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{SmallVec, Tiling};

use crate::{Decline, Dimension, Space, StoragePartitioning, TileMisfit};

use crate::Coords;

/// One operand's physical `(extent, stride)` per axis, in scalars, and its storage tiling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Geometry {
    shape: Vec<usize>,
    strides: Vec<usize>,
    tiling: Tiling,
}

impl Geometry {
    /// One `(extent, stride)` per physical axis, coarsest first, both in scalars.
    pub fn new(axes: &[(usize, usize)]) -> Self {
        Self {
            shape: axes.iter().map(|&(extent, _)| extent).collect(),
            strides: axes.iter().map(|&(_, stride)| stride).collect(),
            tiling: Tiling::UNTILED,
        }
    }

    /// A dense row-major buffer of `shape`, coarsest first: what an output with no tensor behind it
    /// is counted in.
    pub fn row_major(shape: &[usize]) -> Self {
        let mut strides = vec![1; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        Self {
            shape: shape.to_vec(),
            strides,
            tiling: Tiling::UNTILED,
        }
    }

    /// Elements from the first it addresses to past its last: its extents' product where its axes
    /// lie back to back, more where a stride pads one apart; zero for a buffer of no element.
    pub fn span(&self) -> usize {
        if self.shape.contains(&0) {
            return 0;
        }
        self.shape
            .iter()
            .zip(&self.strides)
            .map(|(extent, stride)| (extent - 1) * stride)
            .sum::<usize>()
            + 1
    }

    /// This geometry with its axes stated as pieces of logical axes, as `tiling` says.
    pub(crate) fn with_tiling(mut self, tiling: Tiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// Which axes are pieces of one logical axis: untiled unless the binding said otherwise.
    pub fn tiling(&self) -> Tiling {
        self.tiling
    }

    /// Its trailing axes, each named by the dimension it is a piece of: the names of its
    /// [`pieces`](Self::pieces).
    ///
    /// # Errors
    ///
    /// As [`pieces`](Self::pieces).
    ///
    /// # Panics
    ///
    /// As [`pieces`](Self::pieces).
    pub fn labels(&self, dimensions: &[Dimension]) -> Result<Vec<Dimension>, Decline> {
        Ok(names_of_pieces(self.pieces(dimensions)?))
    }

    /// Its trailing axes, each the dimension it is a piece of and how many of that dimension's
    /// pieces are coarser. `dimensions` name the trailing logical axes in the tensor's order: each
    /// is as many axes as the tiling stores it as, wherever it lists them. The axes ahead of them
    /// are batch axes, one each. An untiled buffer is named axis by axis, a dimension that names
    /// several its coarsest piece first.
    ///
    /// # Errors
    ///
    /// [`Decline::RankBelowLabels`] for a buffer that stands for fewer logical axes than
    /// `dimensions` name, and [`Decline::BatchAxesInPieces`] for a tiling that stores a batch axis
    /// anywhere but ahead, one axis.
    ///
    /// # Panics
    ///
    /// When the tiling does not fit the rank.
    pub fn pieces(&self, dimensions: &[Dimension]) -> Result<Vec<(Dimension, usize)>, Decline> {
        if !self.tiling.is_tiled() {
            return Ok(pieces_named(dimensions));
        }
        let logical_rank = self.tiled_logical_rank();
        let batches =
            logical_rank
                .checked_sub(dimensions.len())
                .ok_or(Decline::RankBelowLabels {
                    rank: logical_rank,
                    block: dimensions.len(),
                })?;
        let axes = self.tiling.physical_dims(logical_rank);
        let ahead = &axes[..batches];
        if !ahead.iter().flatten().copied().eq(0..batches) {
            return Err(Decline::BatchAxesInPieces(self.tiling));
        }
        let named_axes = &self.tiled_axes()[batches..];
        let name = |&(logical, piece): &(usize, usize)| (dimensions[logical - batches], piece);
        Ok(named_axes.iter().map(name).collect())
    }

    /// For each of its trailing axes, which `labels` name, how many of the named dimension's pieces
    /// are coarser.
    pub(crate) fn pieces_of_axes(
        &self,
        labels: &[Dimension],
    ) -> SmallVec<[usize; Space::MAX_RANK]> {
        match self.tiling.is_tiled() {
            true => self.pieces_of_tiled_axes(labels.len()),
            false => pieces_named(labels)
                .into_iter()
                .map(|(_, piece)| piece)
                .collect(),
        }
    }

    /// [`pieces_of_axes`](Self::pieces_of_axes) of the trailing `labelled` axes of a tiled buffer.
    fn pieces_of_tiled_axes(&self, labelled: usize) -> SmallVec<[usize; Space::MAX_RANK]> {
        let unlabelled = self.rank().saturating_sub(labelled);
        self.tiled_axes()[unlabelled..]
            .iter()
            .map(|&(_, piece)| piece)
            .collect()
    }

    /// For each axis of a tiled buffer, the logical axis it is a piece of and how many of that
    /// logical axis's pieces are coarser.
    fn tiled_axes(&self) -> SmallVec<[(usize, usize); Space::MAX_RANK]> {
        let mut axes = SmallVec::from_elem((0, 0), self.rank());
        let logical_axes = self.tiling.physical_dims(self.tiled_logical_rank());
        for (logical, pieces) in logical_axes.into_iter().enumerate() {
            for (piece, axis) in pieces.into_iter().enumerate() {
                axes[axis] = (logical, piece);
            }
        }
        axes
    }

    /// How many logical axes a tiled buffer stands for.
    fn tiled_logical_rank(&self) -> usize {
        self.tiling
            .logical_rank(self.rank())
            .unwrap_or_else(|e| panic!("Geometry: the tiling does not fit the rank: {e:?}"))
    }

    /// The extents, coarsest first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides that step them, in scalars.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// How many physical axes the operand has.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// This geometry with its trailing `labels.len()` axes and labels reordered by stride, coarsest
    /// first; leading (batch) axes stay put.
    pub fn in_stride_order(&self, labels: &[Dimension]) -> (Geometry, Vec<Dimension>) {
        let batch_axes = self.rank() - labels.len();
        let mut trailing: Vec<(usize, (usize, usize))> =
            self.axes().enumerate().skip(batch_axes).collect();
        trailing.sort_by_key(|&(_, axis)| stride_order(axis));
        let axes: Vec<(usize, usize)> = self
            .axes()
            .take(batch_axes)
            .chain(trailing.iter().map(|&(_, axis)| axis))
            .collect();
        let ordered = trailing
            .iter()
            .map(|&(axis, _)| labels[axis - batch_axes])
            .collect();
        (Geometry::new(&axes), ordered)
    }

    pub fn axes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.shape.iter().copied().zip(self.strides.iter().copied())
    }

    /// Whether this buffer can be read a `tile` at a time, `tile` finest first over the dimensions
    /// `labels` name, right-aligned to the axes.
    pub fn serves(
        &self,
        tile: &[(Dimension, usize)],
        labels: &[Dimension],
    ) -> Result<(), LineMisfit> {
        let values: usize = tile.iter().map(|&(_, count)| count).product();
        if values == 1 {
            return Ok(());
        }
        if self.rank() == 0 {
            return Err(LineMisfit::NoAxes);
        }
        let storage = StoragePartitioning::new(self, labels).ok_or(LineMisfit::NotPartitioned)?;
        storage
            .holds(tile, &self.extents(labels))
            .map_err(LineMisfit::Tile)?;
        // The run, from the finest labelled axis up: each axis it takes steps by what it has taken
        // so far. An axis no label names holds no value of the tile.
        let unlabelled = self.rank().saturating_sub(labels.len());
        let mut by_stride: SmallVec<[(usize, (usize, usize)); 8]> = self
            .axes()
            .enumerate()
            .skip(unlabelled)
            .filter(|&(_, (extent, stride))| extent > 1 && stride > 0)
            .collect();
        by_stride.sort_by_key(|&(axis, (_, stride))| (stride, core::cmp::Reverse(axis)));
        let mut inside: SmallVec<[usize; 8]> = SmallVec::new();
        let mut run = 1;
        for &(axis, (extent, stride)) in &by_stride {
            if run >= values {
                break;
            }
            if stride != run {
                return Err(match run {
                    1 => LineMisfit::InnermostStrided(stride),
                    _ => LineMisfit::Gap(stride),
                });
            }
            inside.push(axis);
            run *= extent;
        }
        // An axis of one element steps nowhere, so its stride lands no value inside a line.
        match self.axes().enumerate().find(|&(axis, (extent, stride))| {
            extent > 1 && !inside.contains(&axis) && !stride.is_multiple_of(values)
        }) {
            Some((_, (_, stride))) => Err(LineMisfit::StrideInsideLine(stride)),
            None => Ok(()),
        }
    }

    /// Each labelled dimension's extent: the product of the axes it labels.
    pub(crate) fn extents(&self, labels: &[Dimension]) -> Vec<(Dimension, usize)> {
        let unlabelled = self.rank().saturating_sub(labels.len());
        let mut extents: Vec<(Dimension, usize)> = Vec::new();
        for (axis, extent) in self.shape.iter().copied().enumerate().skip(unlabelled) {
            let dimension = labels[axis - unlabelled];
            match extents.iter_mut().find(|(a, _)| *a == dimension) {
                Some((_, product)) => *product *= extent,
                None => extents.push((dimension, extent)),
            }
        }
        extents
    }

    /// Elements that lie back to back from the first: the axes taken finest stride first, each
    /// while it steps by what the finer ones hold. A stride padded past that run ends it.
    pub(crate) fn dense_run(&self) -> usize {
        let mut run = 1;
        for (_, (extent, stride)) in self.addressing_axes_by_stride().into_iter().rev() {
            if stride != run {
                break;
            }
            run *= extent;
        }
        run
    }

    /// The axes with extent above one and non-zero stride, as `(index, (extent, stride))`,
    /// coarsest stride first.
    pub(crate) fn addressing_axes_by_stride(
        &self,
    ) -> SmallVec<[(usize, (usize, usize)); Space::MAX_RANK]> {
        let mut axes: SmallVec<[(usize, (usize, usize)); Space::MAX_RANK]> = self
            .axes()
            .enumerate()
            .filter(|&(_, (extent, stride))| extent > 1 && stride > 0)
            .collect();
        axes.sort_by_key(|&(_, (_, stride))| core::cmp::Reverse(stride));
        axes
    }
}

/// Why a [`Geometry`] cannot be read a tile at a time ([`Geometry::serves`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineMisfit {
    /// The finest axis's own stride, when it is not 1: consecutive values are not one run.
    InnermostStrided(usize),
    /// An axis inside the tile that does not step by what is finer than it: a gap in the run.
    Gap(usize),
    /// The storage partitioning does not hold the tile: see [`TileMisfit`].
    Tile(TileMisfit),
    /// A stride outside the tile that is not a whole number of tiles.
    StrideInsideLine(usize),
    /// The buffer's order is no partitioning: a stated piece coarser than the rest of a dimension.
    NotPartitioned,
    /// No axes at all, so only a single value fits.
    NoAxes,
}

impl Display for LineMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InnermostStrided(stride) => write!(
                f,
                "its finest axis steps by {stride} rather than 1, so consecutive values are not \
                 one run"
            ),
            Self::Gap(stride) => write!(
                f,
                "an axis inside the tile steps by {stride}, leaving a gap in its run"
            ),
            Self::Tile(why) => write!(f, "its storage does not hold the tile: {why}"),
            Self::StrideInsideLine(stride) => write!(
                f,
                "its stride {stride} is not a whole number of tiles, so a coarser step lands \
                 inside one"
            ),
            Self::NotPartitioned => write!(
                f,
                "a stated piece sits coarser than the rest of a dimension, so its order is no \
                 partitioning"
            ),
            Self::NoAxes => write!(f, "it has no axes, so only a single value fits"),
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
    /// An empty geometry, grown an axis at a time by `push`.
    // A cube constructor: `Default` has no kernel expansion.
    #[allow(clippy::new_without_default)]
    pub fn new() -> RuntimeGeometry {
        RuntimeGeometry {
            shape: Coords::<u32>::new(),
            strides: Coords::<u32>::new(),
        }
    }

    /// The geometry a launched tensor carries, over its first `rank` axes.
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

    /// One physical axis, coarsest first: how far it runs, and the scalar stride that steps it.
    pub fn push(&mut self, extent: u32, stride: u32) {
        self.shape.push(extent);
        self.strides.push(stride);
    }
}

/// The dimension each of `pieces` is a piece of.
pub(crate) fn names_of_pieces(
    pieces: impl IntoIterator<Item = (Dimension, usize)>,
) -> Vec<Dimension> {
    pieces.into_iter().map(|(dimension, _)| dimension).collect()
}

/// The [`pieces`](Geometry::pieces) of an untiled buffer's axes, which `labels` name.
pub(crate) fn pieces_named(labels: &[Dimension]) -> Vec<(Dimension, usize)> {
    let named = |(axis, &label): (usize, &Dimension)| {
        let coarser = labels[..axis].iter().filter(|&&other| other == label);
        (label, coarser.count())
    };
    labels.iter().enumerate().map(named).collect()
}

/// Where an axis of `(extent, stride)` sorts among a buffer's axes in stride order: by stride,
/// coarsest first, and on a tie an axis of one element before the others. It steps nowhere, so its
/// stride places it nowhere: a column `[k, 1]` at strides `[1, 1]` runs along `k`, and its lines
/// run along `k` too, not along the axis one element long.
pub(crate) fn stride_order((extent, stride): (usize, usize)) -> (core::cmp::Reverse<usize>, bool) {
    (core::cmp::Reverse(stride), extent != 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageLevels;

    const M: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

    /// A geometry's span is the elements its strides reach, padding included, and zero when it
    /// holds none.
    #[test]
    fn a_span_reaches_past_the_last_element_its_strides_address() {
        assert_eq!(Geometry::row_major(&[3, 4]).span(), 12);
        assert_eq!(Geometry::new(&[(3, 5), (4, 1)]).span(), 14);
        assert_eq!(Geometry::new(&[(0, 5), (4, 1)]).span(), 0);
    }

    /// A line along a labelled axis takes no step of an axis no label names.
    #[test]
    fn an_unlabelled_axis_holds_no_value_of_a_line() {
        let batched = Geometry::new(&[(8, 1), (64, 8)]);
        assert_eq!(
            batched.serves(&[(N, 4)], &[N]),
            Err(LineMisfit::InnermostStrided(8))
        );
    }

    #[test]
    fn stride_order_reorders_the_axes_and_their_labels() {
        // `[n, k]` behind a `[k, n]` view: `k` strides by one.
        let (axes, labels) = Geometry::new(&[(4096, 1), (6144, 4096)]).in_stride_order(&[K, N]);
        assert_eq!(labels, vec![N, K]);
        assert_eq!(axes, Geometry::new(&[(6144, 4096), (4096, 1)]));

        // Already in order: nothing moves.
        let (axes, labels) = Geometry::new(&[(4096, 6144), (6144, 1)]).in_stride_order(&[K, N]);
        assert_eq!(labels, vec![K, N]);
        assert_eq!(axes, Geometry::new(&[(4096, 6144), (6144, 1)]));

        // A broadcast batch axis strides by zero and stays first all the same.
        let (axes, labels) =
            Geometry::new(&[(8, 0), (4096, 1), (6144, 4096)]).in_stride_order(&[K, N]);
        assert_eq!(labels, vec![N, K]);
        assert_eq!(axes, Geometry::new(&[(8, 0), (6144, 4096), (4096, 1)]));
    }

    /// On untiled buffers, `serves` agrees with the one-axis line rule ([`line_rule_serves`]).
    #[test]
    fn a_line_is_served_where_the_line_rule_served_it() {
        let geometries: Vec<(Geometry, Vec<Dimension>)> = vec![
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
            // A row viewed out of a column: its single row keeps the column's stride.
            (Geometry::new(&[(1, 1), (8, 1)]), vec![M, N]),
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
        let labels = geometry.labels(&[K, N]).unwrap();
        assert_eq!(geometry.serves(&[(N, 4)], &labels), Ok(()));
        assert_eq!(geometry.serves(&[(N, 32), (K, 32)], &labels), Ok(()));
        assert!(geometry.serves(&[(N, 2)], &labels).is_err());
        assert!(geometry.serves(&[(N, 8)], &labels).is_err());
    }

    /// Each axis is the piece the tiling states, behind batch axes of their own.
    #[test]
    fn pieces_are_those_the_tiling_states() {
        let tiling = Tiling::from_physical_dims(&[&[0][..], &[1, 5, 3], &[2, 4, 6]]).unwrap();
        let geometry = Geometry::row_major(&[7, 3, 5, 8, 8, 2, 2]).with_tiling(tiling);
        assert_eq!(
            geometry.pieces(&[M, K]),
            Ok(vec![(M, 0), (K, 0), (M, 2), (K, 1), (M, 1), (K, 2)])
        );
    }

    #[test]
    fn a_batch_axis_among_the_pieces_is_declined() {
        let tiling = Tiling::from_physical_dims(&[&[1][..], &[0, 3], &[2, 4]]).unwrap();
        let geometry = Geometry::row_major(&[3, 7, 5, 8, 8]).with_tiling(tiling);
        assert_eq!(
            geometry.pieces(&[M, K]),
            Err(Decline::BatchAxesInPieces(tiling))
        );
    }

    /// With no dimension named, every logical axis is a batch axis, and a tiled one is stored as
    /// several axes.
    #[test]
    fn a_tiled_buffer_named_by_no_dimension_is_declined() {
        let tiling = Tiling::from_physical_dims(&[&[0][..], &[1, 2]]).unwrap();
        let geometry = Geometry::row_major(&[3, 5, 8]).with_tiling(tiling);
        assert_eq!(
            geometry.pieces(&[]),
            Err(Decline::BatchAxesInPieces(tiling))
        );
    }

    #[test]
    fn more_dimensions_than_a_tiled_buffer_stands_for_are_declined() {
        let geometry = Geometry::row_major(&[5, 8])
            .with_tiling(Tiling::new(&[2]).expect("two pieces should be a tiling"));
        assert_eq!(
            geometry.pieces(&[M, K]),
            Err(Decline::RankBelowLabels { rank: 1, block: 2 })
        );
    }

    /// The one-axis line rule: a line of `width` is served where the innermost axis is contiguous
    /// and its extent and every outer stride are multiples of `width`; an outer axis of one
    /// element steps nowhere, so its stride counts for nothing.
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
                .zip(geometry.shape())
                .all(|(s, &extent)| extent == 1 || s.is_multiple_of(width))
    }
}
