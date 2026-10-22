//! [`StoragePartitioning`]: how a tensor's values are laid down in memory, as nested tiles.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{MetadataError, SmallVec, Tiling};

use crate::{Axis, Geometry, Space};

/// The most pieces a statement keeps on the stack: a few stated tiles and the rest of each axis.
const PIECES: usize = 12;

/// A list of pieces, or of what each piece gives, without a heap allocation.
type PieceList<T> = SmallVec<[T; PIECES]>;

/// How a tensor's values are laid down: stated tiles, finest first, then the rest's order.
///
/// ```ignore
/// // 32 x 32 tiles of a [k, n] weight, stored for a four-wide read, the tiles along N
/// let storage = StorageLevels::new(&[(N, 4)])
///     .tile(&[(N, 8), (K, 32)])
///     .grid(&[N, K]);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct StoragePartitioning {
    /// Stated tiles, finest first: each `(axis, count)` holds `count` of the piece below it.
    tiles: PieceList<(Axis, usize)>,
    /// The rest of each axis, finest first; its counts are what the extents leave.
    order: SmallVec<[Axis; Space::MAX_RANK]>,
}

impl StoragePartitioning {
    /// The statement a bound buffer was written from, read off its dims by stride; `labels` name
    /// the trailing dims. `None` where the buffer's order is no partitioning.
    pub fn new(geometry: &Geometry, labels: &[Axis]) -> Option<Self> {
        let rank = geometry.rank();
        let unlabelled = rank.saturating_sub(labels.len());
        let tiling = geometry.tiling();
        // Dims past the logical rank are stated pieces.
        let first_stated = match tiling.is_tiled() {
            true => tiling.logical_rank(rank).unwrap_or(rank),
            false => rank,
        };
        let mut dims: PieceList<(usize, usize, usize)> = geometry
            .dims()
            .enumerate()
            .filter(|&(dim, (extent, stride))| {
                let piece_of_one = extent == 1 && dim >= first_stated;
                dim >= unlabelled && (piece_of_one || extent > 1 && stride > 0)
            })
            .map(|(dim, (extent, stride))| (dim, extent, stride))
            .collect();
        dims.sort_by_key(|&(dim, extent, stride)| (stride, extent > 1, core::cmp::Reverse(dim)));
        let (stated, rest): (PieceList<_>, PieceList<_>) = dims
            .iter()
            .copied()
            .partition(|&(dim, _, _)| dim >= first_stated);
        let finest_rest = rest.first().map_or(usize::MAX, |&(_, _, stride)| stride);
        if stated
            .iter()
            .any(|&(_, extent, stride)| extent > 1 && stride > finest_rest)
        {
            return None;
        }
        Some(Self {
            tiles: stated
                .iter()
                .map(|&(dim, extent, _)| (labels[dim - unlabelled], extent))
                .collect(),
            order: rest
                .iter()
                .map(|&(dim, _, _)| labels[dim - unlabelled])
                .collect(),
        })
    }

    /// The stated tiles, finest first.
    pub fn tiles(&self) -> &[(Axis, usize)] {
        &self.tiles
    }

    /// The order the rest of each axis follows, finest first.
    pub fn order(&self) -> &[Axis] {
        &self.order
    }

    /// The buffer's dims, coarsest first, each named by the axis it is a piece of.
    pub fn labels(&self, axes: &[Axis]) -> Vec<Axis> {
        StoragePartitioning::level_major(axes, &self.fragments(axes))
    }

    /// The dims of a buffer storing `axes` as `fragments` pieces each, in cubecl's [`Tiling`]
    /// order: `[k, n]` stored `[k/32, n/32, 32, 32]` is `[K, N, K, N]`.
    pub(crate) fn level_major(axes: &[Axis], fragments: &[usize]) -> Vec<Axis> {
        assert_eq!(
            axes.len(),
            fragments.len(),
            "StoragePartitioning::level_major: {} axes but {} piece counts",
            axes.len(),
            fragments.len()
        );
        let depth = fragments.iter().copied().max().unwrap_or(0);
        (0..depth)
            .flat_map(|level| {
                axes.iter()
                    .zip(fragments)
                    .filter(move |&(_, &pieces)| level < pieces)
                    .map(|(&axis, _)| axis)
            })
            .collect()
    }

    /// Whether `tile`, finest first, is held by this partitioning over a tensor of `extents`.
    /// Stated tiles are taken whole; the rest of an axis may be divided.
    pub fn holds(
        &self,
        tile: &[(Axis, usize)],
        extents: &[(Axis, usize)],
    ) -> Result<(), TileMisfit> {
        let mut pieces = self.pieces(extents)?;
        pieces.reverse();
        for &(axis, count) in tile {
            let mut run = 1;
            while run < count {
                let Some(mut piece) = pieces.pop() else {
                    return Err(TileMisfit::RunsOut { wanted: axis });
                };
                if piece.count == 1 {
                    continue;
                }
                if piece.axis != axis {
                    return Err(TileMisfit::Interleaved {
                        wanted: axis,
                        found: piece.axis,
                    });
                }
                let missing = count / run;
                if missing.is_multiple_of(piece.count) {
                    run *= piece.count;
                    continue;
                }
                if piece.stated || !piece.count.is_multiple_of(missing) {
                    return Err(TileMisfit::Overshoots {
                        axis,
                        wanted: count,
                        piece: piece.count,
                    });
                }
                piece.count /= missing;
                pieces.push(piece);
                run = count;
            }
        }
        Ok(())
    }

    /// The tiles, finest first, a window can be addressed inside with one stride per axis.
    pub(crate) fn contiguous_tiles(
        &self,
        extents: &[(Axis, usize)],
    ) -> PieceList<SmallVec<[(Axis, usize); Space::MAX_RANK]>> {
        let Ok(pieces) = self.pieces(extents) else {
            return PieceList::new();
        };
        let mut tiles = PieceList::new();
        let mut tile: SmallVec<[(Axis, usize); Space::MAX_RANK]> = SmallVec::new();
        for piece in pieces.into_iter().filter(|piece| piece.count > 1) {
            let continues = tile.last().is_some_and(|&(axis, _)| axis == piece.axis);
            match tile.iter_mut().find(|(axis, _)| *axis == piece.axis) {
                // An axis met again after another one sat between: no longer one run per axis.
                Some(_) if !continues => break,
                Some((_, extent)) => *extent *= piece.count,
                None => tile.push((piece.axis, piece.count)),
            }
            tiles.push(tile.clone());
        }
        tiles
    }

    /// The buffer storing a tensor of `extents`: its dims in [`labels`](Self::labels)' order and
    /// its [`Tiling`].
    pub fn physical(&self, extents: &[(Axis, usize)]) -> Result<Geometry, StorageMisfit> {
        let axes: SmallVec<[Axis; Space::MAX_RANK]> =
            extents.iter().map(|&(axis, _)| axis).collect();
        let named = |axis: &Axis| self.order.iter().filter(|&a| a == axis).count() == 1;
        if self.order.len() != axes.len() || !axes.iter().all(named) {
            return Err(StorageMisfit::Order {
                order: self.order.to_vec(),
                axes: axes.to_vec(),
            });
        }
        let pieces = self.pieces(extents).map_err(|misfit| match misfit {
            TileMisfit::Unwhole { axis, extent, tile } => {
                StorageMisfit::PartialTile { axis, extent, tile }
            }
            _ => unreachable!("only the rest of an axis is computed against its extent"),
        })?;
        let tiling = Tiling::new(&self.fragments(&axes)).map_err(StorageMisfit::Unrecordable)?;
        let strides: PieceList<usize> = pieces
            .iter()
            .scan(1, |run, piece| {
                let stride = *run;
                *run *= piece.count;
                Some(stride)
            })
            .collect();
        // `labels` names an axis's pieces coarsest first; `pieces` lists them finest first.
        let mut taken: PieceList<bool> = SmallVec::from_elem(false, pieces.len());
        let dims: PieceList<(usize, usize)> = self
            .labels(&axes)
            .iter()
            .map(|&axis| {
                let p = (0..pieces.len())
                    .rev()
                    .find(|&p| !taken[p] && pieces[p].axis == axis)
                    .expect("an axis has as many pieces as it has dims");
                taken[p] = true;
                (pieces[p].count, strides[p])
            })
            .collect();
        Ok(Geometry::new(&dims).with_tiling(tiling))
    }

    /// How many buffer dims each of `axes` is stored as: one per stated tile along it, and one for
    /// the rest of it.
    fn fragments(&self, axes: &[Axis]) -> SmallVec<[usize; Space::MAX_RANK]> {
        axes.iter()
            .map(|&axis| 1 + self.tiles.iter().filter(|&&(a, _)| a == axis).count())
            .collect()
    }

    /// Every piece over a tensor of `extents`, finest first: stated tiles, then each rest.
    fn pieces(&self, extents: &[(Axis, usize)]) -> Result<PieceList<Piece>, TileMisfit> {
        let mut pieces: PieceList<Piece> = self
            .tiles
            .iter()
            .map(|&(axis, count)| Piece {
                axis,
                count,
                stated: true,
            })
            .collect();
        for &axis in &self.order {
            let extent = extents
                .iter()
                .find(|&&(a, _)| a == axis)
                .map_or(1, |&(_, extent)| extent);
            let tile: usize = self
                .tiles
                .iter()
                .filter(|&&(a, _)| a == axis)
                .map(|&(_, count)| count)
                .product();
            if !extent.is_multiple_of(tile) {
                return Err(TileMisfit::Unwhole { axis, extent, tile });
            }
            pieces.push(Piece {
                axis,
                count: extent / tile,
                stated: false,
            });
        }
        Ok(pieces)
    }
}

/// A [`StoragePartitioning`] being stated leaf-up, until [`grid`](Self::grid).
#[derive(Clone, Debug)]
pub struct StorageLevels {
    tiles: PieceList<(Axis, usize)>,
}

impl StorageLevels {
    /// The finest tile, `(axis, count)` finest first, in values.
    pub fn new(tile: &[(Axis, usize)]) -> Self {
        Self {
            tiles: SmallVec::from_slice(tile),
        }
    }

    /// A coarser tile, each count how many of the tile below it holds.
    pub fn tile(mut self, level: &[(Axis, usize)]) -> Self {
        self.tiles.extend_from_slice(level);
        self
    }

    /// The partitioning, the rest of each axis following `order`, finest first.
    pub fn grid(self, order: &[Axis]) -> StoragePartitioning {
        StoragePartitioning {
            tiles: self.tiles,
            order: SmallVec::from_slice(order),
        }
    }
}

/// Where a tile stops being held by a [`StoragePartitioning`] ([`StoragePartitioning::holds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileMisfit {
    /// A piece of another axis sits inside a wanted piece: the wanted tile is not one run.
    Interleaved { wanted: Axis, found: Axis },
    /// A piece reaches past the wanted count.
    Overshoots {
        axis: Axis,
        wanted: usize,
        piece: usize,
    },
    /// The partitioning ends before the wanted piece closes.
    RunsOut { wanted: Axis },
    /// The stated tiles do not close the axis's extent.
    Unwhole {
        axis: Axis,
        extent: usize,
        tile: usize,
    },
}

impl Display for TileMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interleaved { wanted, found } => write!(
                f,
                "a piece along {found:?} sits inside the one wanted along {wanted:?}"
            ),
            Self::Overshoots {
                axis,
                wanted,
                piece,
            } => write!(
                f,
                "a piece of {piece} along {axis:?} reaches past the {wanted} wanted, and is not \
                 the rest of the axis, which it would divide"
            ),
            Self::RunsOut { wanted } => {
                write!(f, "nothing is left along {wanted:?} for the piece wanted")
            }
            Self::Unwhole { axis, extent, tile } => write!(
                f,
                "axis {axis:?} runs {extent}, which is not a whole number of {tile}-wide tiles"
            ),
        }
    }
}

/// Why a [`StoragePartitioning`] cannot store a tensor ([`StoragePartitioning::physical`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageMisfit {
    /// The tiles along `axis` do not close its extent.
    PartialTile {
        axis: Axis,
        extent: usize,
        tile: usize,
    },
    /// The order does not name each of the tensor's axes once.
    Order { order: Vec<Axis>, axes: Vec<Axis> },
    /// More pieces than the buffer's metadata records ([`Tiling`]).
    Unrecordable(MetadataError),
}

impl Display for StorageMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartialTile { axis, extent, tile } => write!(
                f,
                "axis {axis:?} runs {extent}, which is not a whole number of {tile}-wide tiles"
            ),
            Self::Order { order, axes } => write!(
                f,
                "the tiles follow {order:?}, but the tensor's axes are {axes:?}: the order names \
                 each once"
            ),
            Self::Unrecordable(why) => {
                write!(f, "the buffer's metadata cannot record its pieces: {why:?}")
            }
        }
    }
}

/// One piece of a partitioning over a tensor: `count` of the piece below it along `axis`.
#[derive(Clone, Copy, Debug)]
struct Piece {
    axis: Axis,
    count: usize,
    /// Whether the count was stated, so a reader must take the piece whole.
    stated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A buffer reads back as the statement it was written from, whatever order its tiling lists.
    #[test]
    fn a_written_buffer_reads_back_as_its_statement() {
        let statements = [
            StorageLevels::new(&[(N, 32), (K, 16)]).grid(&[N, K]),
            StorageLevels::new(&[(N, 16), (K, 32)]).grid(&[K, N]),
            StorageLevels::new(&[(N, 4), (N, 8)])
                .tile(&[(K, 2), (K, 16)])
                .grid(&[K, N]),
            StorageLevels::new(&[(N, 4)])
                .tile(&[(N, 1), (K, 2)])
                .grid(&[N, K]),
        ];
        for storage in statements {
            let geometry = storage.physical(&[(K, 64), (N, 64)]).unwrap();
            let read = StoragePartitioning::new(&geometry, &geometry.labels(&[K, N])).unwrap();
            assert_eq!(read, storage);
        }
    }

    /// An untiled buffer states no tiles: its order is its dims by stride.
    #[test]
    fn an_untiled_buffer_is_its_order_alone() {
        let rows = Geometry::new(&[(64, 4096), (4096, 1)]);
        let read = StoragePartitioning::new(&rows, &[K, N]).unwrap();
        assert!(read.tiles().is_empty());
        assert_eq!(read.order(), &[N, K]);

        let transposed = Geometry::new(&[(64, 1), (4096, 64)]);
        let read = StoragePartitioning::new(&transposed, &[K, N]).unwrap();
        assert_eq!(read.order(), &[K, N]);

        let broadcast = Geometry::new(&[(3, 0), (1, 5), (4096, 1)]);
        let read = StoragePartitioning::new(&broadcast, &[M, K, N]).unwrap();
        assert_eq!(read.order(), &[N]);
    }

    #[test]
    fn a_stored_read_and_its_whole_tile_are_both_held() {
        let with_read = StorageLevels::new(&[(N, 4)])
            .tile(&[(N, 8), (K, 32)])
            .grid(&[N, K]);
        let extents = [(K, 64), (N, 64)];
        assert_eq!(with_read.holds(&[(N, 4)], &extents), Ok(()));
        assert_eq!(with_read.holds(&[(N, 32), (K, 32)], &extents), Ok(()));
    }

    #[test]
    fn a_stated_tile_is_never_split() {
        let extents = [(K, 64), (N, 64)];
        let tiles = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K]);
        assert_eq!(
            tiles.holds(&[(N, 4)], &extents),
            Err(TileMisfit::Overshoots {
                axis: N,
                wanted: 4,
                piece: 32
            })
        );
        let column_first = StorageLevels::new(&[(K, 32), (N, 32)]).grid(&[N, K]);
        assert_eq!(
            column_first.holds(&[(N, 32), (K, 32)], &extents),
            Err(TileMisfit::Interleaved {
                wanted: N,
                found: K
            })
        );
    }

    #[test]
    fn a_read_cuts_itself_out_of_the_rest_of_an_axis() {
        let rows = StorageLevels::new(&[]).grid(&[N, K]);
        let extents = [(K, 64), (N, 4096)];
        assert_eq!(rows.holds(&[(N, 8)], &extents), Ok(()));
        assert!(rows.holds(&[(N, 3)], &extents).is_err());
    }

    /// NVFP4: a word of eight along `K` and a two-by-two-word read are whole stated tiles.
    #[test]
    fn a_word_and_a_two_dimensional_read_are_whole_tiles() {
        let values = StorageLevels::new(&[(K, 8)])
            .tile(&[(K, 2), (N, 2)])
            .tile(&[(K, 8), (N, 8)])
            .grid(&[K, N]);
        let extents = [(K, 256), (N, 64)];
        assert_eq!(values.holds(&[(K, 8)], &extents), Ok(()));
        assert_eq!(values.holds(&[(K, 16), (N, 2)], &extents), Ok(()));
        assert!(values.holds(&[(K, 32)], &extents).is_err());
    }

    #[test]
    fn a_k_first_grid_moves_the_strides_not_the_dims() {
        let storage = StorageLevels::new(&[(N, 16), (K, 32)]).grid(&[K, N]);
        let geometry = storage.physical(&[(K, 128), (N, 64)]).unwrap();
        // [k/32, n/16, 32, 16]: a tile is 512 values, the next along K 512 on, along N 4 tiles on.
        assert_eq!(
            geometry,
            Geometry::new(&[(4, 512), (4, 2048), (32, 16), (16, 1)])
                .with_tiling(cubecl::zspace::Tiling::new(&[2, 2]).unwrap())
        );
    }

    #[test]
    fn a_statement_that_does_not_fit_the_tensor_is_refused_when_written() {
        let eight_rows = StorageLevels::new(&[(K, 8)]).grid(&[K, N]);
        assert_eq!(
            eight_rows.physical(&[(K, 12), (N, 4)]),
            Err(StorageMisfit::PartialTile {
                axis: K,
                extent: 12,
                tile: 8
            })
        );
        let one_axis = StorageLevels::new(&[(K, 4)]).grid(&[K]);
        assert!(matches!(
            one_axis.physical(&[(K, 12), (N, 4)]),
            Err(StorageMisfit::Order { .. })
        ));
    }

    /// A tile's piece of one is kept as a dim of the buffer, and a reader is held the same either way.
    #[test]
    fn a_piece_of_one_is_a_dim_kept_as_stated() {
        let with_one = StorageLevels::new(&[(N, 4)])
            .tile(&[(N, 1), (K, 2)])
            .grid(&[N, K]);
        let without = StorageLevels::new(&[(N, 4), (K, 2)]).grid(&[N, K]);
        assert_ne!(with_one, without);
        assert_eq!(with_one.labels(&[K, N]), vec![K, N, K, N, N]);
        assert_eq!(without.labels(&[K, N]), vec![K, N, K, N]);
        let (read, extents) = ([(N, 4), (K, 2)], [(K, 8), (N, 8)]);
        assert_eq!(with_one.holds(&read, &extents), Ok(()));
        assert_eq!(without.holds(&read, &extents), Ok(()));
    }

    /// The level-major order cubecl's tiling lists dims in: every axis's coarsest piece in the
    /// tensor's order, then the next of every axis still split, so an untiled axis drops out after
    /// the first level and a deeper axis appears alone at the finest.
    #[test]
    fn labels_name_the_dims_level_major_coarsest_first() {
        let untiled = StorageLevels::new(&[]).grid(&[N, K]);
        assert_eq!(untiled.labels(&[K, N]), vec![K, N]);
        let one_level = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K]);
        assert_eq!(one_level.labels(&[K, N]), vec![K, N, K, N]);
        let deeper_n = StorageLevels::new(&[(N, 4), (N, 8), (K, 32)]).grid(&[N, K]);
        assert_eq!(deeper_n.labels(&[K, N]), vec![K, N, K, N, N]);
        // A batch axis ahead of the tiled block is its own single dim.
        let batched = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K, M]);
        assert_eq!(batched.labels(&[M, K, N]), vec![M, K, N, K, N]);
    }

    /// The windows `storage` addresses with one stride per axis over `extents`, as lists.
    fn windows(
        storage: &StoragePartitioning,
        extents: &[(Axis, usize)],
    ) -> Vec<Vec<(Axis, usize)>> {
        storage
            .contiguous_tiles(extents)
            .iter()
            .map(|tile| tile.to_vec())
            .collect()
    }

    #[test]
    fn a_window_is_one_run_inside_the_tiles_whose_axes_do_not_interleave() {
        let with_read = StorageLevels::new(&[(N, 4)])
            .tile(&[(N, 8), (K, 32)])
            .grid(&[N, K]);
        assert_eq!(
            windows(&with_read, &[(K, 64), (N, 64)]),
            vec![vec![(N, 4)], vec![(N, 32)], vec![(N, 32), (K, 32)]]
        );

        let down_k = StorageLevels::new(&[(N, 16), (K, 32)]).grid(&[K, N]);
        assert_eq!(
            windows(&down_k, &[(K, 128), (N, 64)]),
            vec![
                vec![(N, 16)],
                vec![(N, 16), (K, 32)],
                vec![(N, 16), (K, 128)]
            ]
        );

        let interleaved = StorageLevels::new(&[(N, 16), (K, 2)])
            .tile(&[(N, 2), (K, 16)])
            .grid(&[N, K]);
        assert_eq!(
            windows(&interleaved, &[(K, 64), (N, 64)]),
            vec![vec![(N, 16)], vec![(N, 16), (K, 2)]]
        );
    }
}
