//! [`StoragePartitioning`]: how a tensor's values are laid down in memory, as nested tiles.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{SmallVec, Tiling};

use super::Stagger;
use crate::{Dimension, Geometry, Refusal, Space};

/// The most pieces a statement keeps on the stack: a few stated tiles and the rest of each dimension.
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
    /// Stated tiles, finest first: each `(dimension, count)` holds `count` of the piece below it.
    tiles: PieceList<(Dimension, usize)>,
    /// The rest of each dimension, finest first; its counts are what the extents leave.
    order: SmallVec<[Dimension; Space::MAX_RANK]>,
    /// How far apart the coarsest pieces start: back to back where `None`.
    stagger: Option<Stagger>,
}

impl StoragePartitioning {
    /// The statement a bound buffer was written from, read off its dims by stride; `labels` name
    /// the trailing dims. `None` where the buffer's order is no partitioning. Its
    /// [`stagger`](Self::stagger) is not read back: a padded coarsest stride orders the pieces as a
    /// dense one does, so the tiles and their order read the same, but a reader asking where runs
    /// are contiguous asks the geometry's strides (`Geometry::dense_run`).
    pub fn new(geometry: &Geometry, labels: &[Dimension]) -> Option<Self> {
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
            stagger: None,
        })
    }

    /// The stated tiles, finest first.
    pub fn tiles(&self) -> &[(Dimension, usize)] {
        &self.tiles
    }

    /// The order the rest of each dimension follows, finest first.
    pub fn order(&self) -> &[Dimension] {
        &self.order
    }

    /// How far apart its coarsest pieces start: back to back where `None`.
    pub fn stagger(&self) -> Option<Stagger> {
        self.stagger
    }

    /// This partitioning with its coarsest pieces laid `stagger` apart.
    pub fn staggered(self, stagger: Stagger) -> Self {
        Self {
            stagger: Some(stagger),
            ..self
        }
    }

    /// The buffer's dims, coarsest first, each named by the dimension it is a piece of.
    pub fn labels(&self, dimensions: &[Dimension]) -> Vec<Dimension> {
        StoragePartitioning::level_major(dimensions, &self.fragments(dimensions))
    }

    /// The dims of a buffer storing `dimensions` as `fragments` pieces each, in cubecl's [`Tiling`]
    /// order: `[k, n]` stored `[k/32, n/32, 32, 32]` is `[K, N, K, N]`.
    pub(crate) fn level_major(dimensions: &[Dimension], fragments: &[usize]) -> Vec<Dimension> {
        assert_eq!(
            dimensions.len(),
            fragments.len(),
            "StoragePartitioning::level_major: {} dimensions but {} piece counts",
            dimensions.len(),
            fragments.len()
        );
        let depth = fragments.iter().copied().max().unwrap_or(0);
        (0..depth)
            .flat_map(|level| {
                dimensions
                    .iter()
                    .zip(fragments)
                    .filter(move |&(_, &pieces)| level < pieces)
                    .map(|(&dimension, _)| dimension)
            })
            .collect()
    }

    /// Whether `tile`, finest first, is held by this partitioning over a tensor of `extents`.
    /// Stated tiles are taken whole; the rest of a dimension may be divided.
    pub fn holds(
        &self,
        tile: &[(Dimension, usize)],
        extents: &[(Dimension, usize)],
    ) -> Result<(), TileMisfit> {
        let mut pieces = self.pieces(extents)?;
        pieces.reverse();
        for &(dimension, count) in tile {
            let mut run = 1;
            while run < count {
                let Some(mut piece) = pieces.pop() else {
                    return Err(TileMisfit::RunsOut { wanted: dimension });
                };
                if piece.count == 1 {
                    continue;
                }
                if piece.dimension != dimension {
                    return Err(TileMisfit::Interleaved {
                        wanted: dimension,
                        found: piece.dimension,
                    });
                }
                let missing = count / run;
                if missing.is_multiple_of(piece.count) {
                    run *= piece.count;
                    continue;
                }
                if piece.stated || !piece.count.is_multiple_of(missing) {
                    return Err(TileMisfit::Overshoots {
                        dimension,
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

    /// The tiles, finest first, a window can be addressed inside with one stride per dimension, its
    /// pieces back to back. A buffer read back from a staggered one can hold a tile across a pad:
    /// the bound geometry's [`dense_run`](Geometry::dense_run) says how far a run really goes.
    pub(crate) fn contiguous_tiles(
        &self,
        extents: &[(Dimension, usize)],
    ) -> PieceList<SmallVec<[(Dimension, usize); Space::MAX_RANK]>> {
        let Ok(pieces) = self.pieces(extents) else {
            return PieceList::new();
        };
        let mut tiles = PieceList::new();
        let mut tile: SmallVec<[(Dimension, usize); Space::MAX_RANK]> = SmallVec::new();
        for piece in pieces.into_iter().filter(|piece| piece.count > 1) {
            let continues = tile
                .last()
                .is_some_and(|&(dimension, _)| dimension == piece.dimension);
            match tile
                .iter_mut()
                .find(|(dimension, _)| *dimension == piece.dimension)
            {
                // A dimension met again after another one sat between: no longer one run per dimension.
                Some(_) if !continues => break,
                Some((_, extent)) => *extent *= piece.count,
                None => tile.push((piece.dimension, piece.count)),
            }
            tiles.push(tile.clone());
        }
        tiles
    }

    /// The buffer storing a tensor of `extents`: its dims in [`labels`](Self::labels)' order and
    /// its [`Tiling`]. Each piece steps by the run of every piece finer than it, the coarsest piece
    /// of more than one by its [`stagger`](Self::stagger)'s pitch over that run where it has one.
    pub fn physical(&self, extents: &[(Dimension, usize)]) -> Result<Geometry, Refusal> {
        let dimensions: SmallVec<[Dimension; Space::MAX_RANK]> =
            extents.iter().map(|&(dimension, _)| dimension).collect();
        let named =
            |dimension: &Dimension| self.order.iter().filter(|&a| a == dimension).count() == 1;
        if self.order.len() != dimensions.len() || !dimensions.iter().all(named) {
            return Err(Refusal::StorageOrderUnmatched {
                order: self.order.to_vec(),
                dimensions: dimensions.to_vec(),
            });
        }
        let pieces = self.pieces(extents).map_err(|misfit| match misfit {
            TileMisfit::Unwhole {
                dimension,
                extent,
                tile,
            } => Refusal::StoredTileUnwhole {
                dimension,
                extent,
                tile,
            },
            _ => unreachable!("only the rest of a dimension is computed against its extent"),
        })?;
        let tiling = Tiling::new(&self.fragments(&dimensions))
            .map_err(Refusal::StoragePiecesUnrecordable)?;
        // A piece of one steps nowhere, so the stagger pads the coarsest piece that steps.
        let staggered = self
            .stagger
            .zip(pieces.iter().rposition(|piece| piece.count > 1));
        let strides: PieceList<usize> = pieces
            .iter()
            .enumerate()
            .scan(1, |run, (p, piece)| {
                let stride = match staggered {
                    Some((stagger, at)) if at == p => stagger.pitch(*run),
                    _ => *run,
                };
                *run = stride * piece.count;
                Some(stride)
            })
            .collect();
        // `labels` names a dimension's pieces coarsest first; `pieces` lists them finest first.
        let mut taken: PieceList<bool> = SmallVec::from_elem(false, pieces.len());
        let dims: PieceList<(usize, usize)> = self
            .labels(&dimensions)
            .iter()
            .map(|&dimension| {
                let p = (0..pieces.len())
                    .rev()
                    .find(|&p| !taken[p] && pieces[p].dimension == dimension)
                    .expect("a dimension has as many pieces as it has dims");
                taken[p] = true;
                (pieces[p].count, strides[p])
            })
            .collect();
        Ok(Geometry::new(&dims).with_tiling(tiling))
    }

    /// How many buffer dims each of `dimensions` is stored as: one per stated tile along it, and one for
    /// the rest of it.
    fn fragments(&self, dimensions: &[Dimension]) -> SmallVec<[usize; Space::MAX_RANK]> {
        dimensions
            .iter()
            .map(|&dimension| 1 + self.tiles.iter().filter(|&&(a, _)| a == dimension).count())
            .collect()
    }

    /// Every piece over a tensor of `extents`, finest first: stated tiles, then each rest.
    fn pieces(&self, extents: &[(Dimension, usize)]) -> Result<PieceList<Piece>, TileMisfit> {
        let mut pieces: PieceList<Piece> = self
            .tiles
            .iter()
            .map(|&(dimension, count)| Piece {
                dimension,
                count,
                stated: true,
            })
            .collect();
        for &dimension in &self.order {
            let extent = extents
                .iter()
                .find(|&&(a, _)| a == dimension)
                .map_or(1, |&(_, extent)| extent);
            let tile: usize = self
                .tiles
                .iter()
                .filter(|&&(a, _)| a == dimension)
                .map(|&(_, count)| count)
                .product();
            if !extent.is_multiple_of(tile) {
                return Err(TileMisfit::Unwhole {
                    dimension,
                    extent,
                    tile,
                });
            }
            pieces.push(Piece {
                dimension,
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
    tiles: PieceList<(Dimension, usize)>,
}

impl StorageLevels {
    /// The finest tile, `(dimension, count)` finest first, in values.
    pub fn new(tile: &[(Dimension, usize)]) -> Self {
        Self {
            tiles: SmallVec::from_slice(tile),
        }
    }

    /// A coarser tile, each count how many of the tile below it holds.
    pub fn tile(mut self, level: &[(Dimension, usize)]) -> Self {
        self.tiles.extend_from_slice(level);
        self
    }

    /// The partitioning, the rest of each dimension following `order`, finest first.
    pub fn grid(self, order: &[Dimension]) -> StoragePartitioning {
        StoragePartitioning {
            tiles: self.tiles,
            order: SmallVec::from_slice(order),
            stagger: None,
        }
    }
}

/// Where a tile stops being held by a [`StoragePartitioning`] ([`StoragePartitioning::holds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileMisfit {
    /// A piece of another dimension sits inside a wanted piece: the wanted tile is not one run.
    Interleaved { wanted: Dimension, found: Dimension },
    /// A piece reaches past the wanted count.
    Overshoots {
        dimension: Dimension,
        wanted: usize,
        piece: usize,
    },
    /// The partitioning ends before the wanted piece closes.
    RunsOut { wanted: Dimension },
    /// The stated tiles do not close the dimension's extent.
    Unwhole {
        dimension: Dimension,
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
                dimension,
                wanted,
                piece,
            } => write!(
                f,
                "a piece of {piece} along {dimension:?} reaches past the {wanted} wanted, and is not \
                 the rest of the dimension, which it would divide"
            ),
            Self::RunsOut { wanted } => {
                write!(f, "nothing is left along {wanted:?} for the piece wanted")
            }
            Self::Unwhole {
                dimension,
                extent,
                tile,
            } => write!(
                f,
                "dimension {dimension:?} runs {extent}, which is not a whole number of {tile}-wide tiles"
            ),
        }
    }
}

/// One piece of a partitioning over a tensor: `count` of the piece below it along `dimension`.
#[derive(Clone, Copy, Debug)]
struct Piece {
    dimension: Dimension,
    count: usize,
    /// Whether the count was stated, so a reader must take the piece whole.
    stated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    const M: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

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

    /// A staggered statement differs from its dense one, pads only its coarsest piece's stride,
    /// and reads back as its tiles and order.
    #[test]
    fn a_staggered_buffer_pads_its_coarsest_stride_alone() {
        let dense = StorageLevels::new(&[(K, 16)])
            .tile(&[(N, 16)])
            .grid(&[K, N]);
        let stagger = Stagger::new(1024, 4).expect("a 4-bit value divides a transaction");
        let staggered = dense.clone().staggered(stagger);
        // The two lay the buffer down differently, so a derivation keyed on one never serves the
        // other.
        assert_ne!(staggered, dense);
        let extents = [(K, 4096), (N, 64)];
        let dense_geometry = dense.physical(&extents).unwrap();
        let geometry = staggered.physical(&extents).unwrap();
        let mut by_stride: Vec<(usize, usize)> = dense_geometry
            .dims()
            .zip(geometry.dims())
            .map(|(d, s)| (d.1, s.1))
            .collect();
        by_stride.sort();
        let (coarsest, finer) = by_stride.split_last().unwrap();
        assert!(finer.iter().all(|&(dense, staggered)| dense == staggered));
        assert_eq!(coarsest.1, stagger.pitch(coarsest.0));
        assert!(coarsest.1 > coarsest.0);
        let read = StoragePartitioning::new(&geometry, &geometry.labels(&[K, N])).unwrap();
        assert_eq!(read, dense);
    }

    /// A rest of one steps nowhere, so the stagger pads the coarsest piece that does: one strip of
    /// tiles along N leaves its tiles down K staggered.
    #[test]
    fn a_stagger_skips_a_coarsest_piece_of_one() {
        let one_strip = StorageLevels::new(&[(K, 16)])
            .tile(&[(N, 32)])
            .grid(&[K, N])
            .staggered(Stagger::new(1024, 4).expect("a 4-bit value divides a transaction"));
        // Each K tile is 512 values, two transactions: three hold it.
        let geometry = one_strip.physical(&[(K, 4096), (N, 32)]).unwrap();
        let k_rest = geometry
            .dims()
            .find(|&(extent, _)| extent == 256)
            .expect("4096 / 16 tiles down K");
        assert_eq!(k_rest.1, 3 * 256);
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
                dimension: N,
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
    fn a_read_cuts_itself_out_of_the_rest_of_a_dimension() {
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
            Err(Refusal::StoredTileUnwhole {
                dimension: K,
                extent: 12,
                tile: 8
            })
        );
        let one_dimension = StorageLevels::new(&[(K, 4)]).grid(&[K]);
        assert!(matches!(
            one_dimension.physical(&[(K, 12), (N, 4)]),
            Err(Refusal::StorageOrderUnmatched { .. })
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

    /// The level-major order cubecl's tiling lists dims in: every dimension's coarsest piece in the
    /// tensor's order, then the next of every dimension still split, so an untiled dimension drops out after
    /// the first level and a deeper dimension appears alone at the finest.
    #[test]
    fn labels_name_the_dims_level_major_coarsest_first() {
        let untiled = StorageLevels::new(&[]).grid(&[N, K]);
        assert_eq!(untiled.labels(&[K, N]), vec![K, N]);
        let one_level = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K]);
        assert_eq!(one_level.labels(&[K, N]), vec![K, N, K, N]);
        let deeper_n = StorageLevels::new(&[(N, 4), (N, 8), (K, 32)]).grid(&[N, K]);
        assert_eq!(deeper_n.labels(&[K, N]), vec![K, N, K, N, N]);
        // A batch dimension ahead of the tiled block is its own single dim.
        let batched = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K, M]);
        assert_eq!(batched.labels(&[M, K, N]), vec![M, K, N, K, N]);
    }

    /// The windows `storage` addresses with one stride per dimension over `extents`, as lists.
    fn windows(
        storage: &StoragePartitioning,
        extents: &[(Dimension, usize)],
    ) -> Vec<Vec<(Dimension, usize)>> {
        storage
            .contiguous_tiles(extents)
            .iter()
            .map(|tile| tile.to_vec())
            .collect()
    }

    #[test]
    fn a_window_is_one_run_inside_the_tiles_whose_dimensions_do_not_interleave() {
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
