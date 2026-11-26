//! Where a stage lives and how it lays its cells out ([`StageStorage`]).

use crate::launch::Delivery;
use crate::{ChunkSwizzle, Decline, Dimension, Instruction, LineBytes, Space, StageForm};
use cubecl::cmma::MatrixLayout;
use cubecl::ir::{MmaProperties, StridedRunRead};
use cubecl::prelude::TensorMapSwizzle;

/// Where a stage lives and how it lays its cells out, stated at
/// [`Stages::smem`](crate::Stages::smem).
///
/// # Assumptions
///
/// - A shared-memory stage lays its axes down in the [`order`](StageOrder) it holds, the
///   innermost one carrying its lines.
/// - A gathered operand stages only into a [`Strided`](StageStorage::Strided) storage in the
///   source's order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    /// Shared memory grouped into `block`-sized tiles, one edge per dimension of the operation.
    Tiled {
        /// The edge of a block along each dimension, read by its dimension (a Keyed Usage).
        block: Vec<(Dimension, usize)>,
        chunks: RowChunks,
        order: StageOrder,
    },
    /// Shared memory as one dense block.
    Strided { order: StageOrder },
    /// The plane's units, one line per unit.
    Lines { read: UnitRead },
}

/// The order in which a shared-memory stage lays its axes down.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StageOrder {
    /// The order of the stage's Space, as the source holds it: its last dimension innermost.
    AsSource,
    /// The order of the stage's Space with this Dimension, the one a contraction sums the operand
    /// along, moved innermost, one value a line: the run of `k` a fragment's lane holds lies
    /// contiguous whatever the source's order.
    AlongRun(Dimension),
}

impl StageOrder {
    /// The order a matrix instruction's fragments read a stage of an operand over `space` in at
    /// full width, the contraction summing it along `contracted`: `contracted` innermost where
    /// the target's fragment load reads a lane's run of `k` laid across its memory's rows one
    /// element at a time ([`StridedRunRead::ElementWise`]) and the source lays it otherwise, the
    /// source's order everywhere else.
    ///
    /// # Guarantees
    ///
    /// - [`AlongRun`](Self::AlongRun) only where both of the target's fragments hold their runs
    ///   along `k` (an `A` along its rows, a `B` along its columns), so the order serves either
    ///   side of the contraction.
    pub(crate) fn for_fragments(space: &Space, contracted: Dimension, mma: &MmaProperties) -> Self {
        let runs_along_k = mma.register_layout_a == MatrixLayout::RowMajor
            && mma.register_layout_b == MatrixLayout::ColMajor;
        let reads_element_wise = mma.strided_run_read == StridedRunRead::ElementWise;
        let innermost = space.dimension_at(space.rank() - 1);
        let reordered = reads_element_wise && runs_along_k && innermost != contracted;
        match reordered && space.contains(contracted) {
            true => StageOrder::AlongRun(contracted),
            false => StageOrder::AsSource,
        }
    }
}

/// How a [`Tiled`](StageStorage::Tiled) block lays each row down, in 16-byte chunks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum RowChunks {
    /// Each row one contiguous run, in order; readable by `cmma` and TMA.
    InOrder,
    /// Each row's chunks XOR-permuted by a row key; read by units addressing their own chunks, and by
    /// a plane group MMA through its descriptor.
    Swizzled,
    /// Each row in order followed by one unused chunk, so rows start on distinct banks.
    Padded,
}

impl RowChunks {
    /// Bytes one chunk holds: the widest read one unit issues.
    pub const CHUNK_BYTES: usize = 16;

    /// Bytes a stage's buffer starts on at least: one pass over the banks, the period the
    /// chunks' swizzle repeats on. A chunk is not enough. The copy engine moves a stage's lines
    /// 16 bytes a unit (`cp.async`), and into a stage that starts mid-sector each copy is split,
    /// so every global sector it reads is fetched twice.
    pub const STAGE_ALIGNMENT_BYTES: usize = 128;

    /// The arrangement a stage asked for as `self` takes where `instruction` reads it and
    /// `delivery` writes it: the one asked for, made one both can use. A fragment API reading
    /// rows off a pointer and a stride (cmma) reads no swizzled row, so a swizzle becomes the
    /// other arrangement that keeps rows on distinct banks, padding; a TMA box lands rows dense,
    /// so padding becomes rows in order.
    pub(crate) fn read_by(self, instruction: Instruction, delivery: Delivery) -> Self {
        let read = match (self, instruction) {
            (Self::Swizzled, Instruction::Cmma) => Self::Padded,
            (chunks, _) => chunks,
        };
        match (read, delivery) {
            (Self::Padded, Delivery::Tma) => Self::InOrder,
            (chunks, _) => chunks,
        }
    }
}

/// How a value reaches the unit asking for it, out of the lines the plane holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum UnitRead {
    /// By plane shuffle: one shuffle per word of the line plus a dynamic pick.
    Shuffle,
    /// Through a plane-owned shared-memory window written once per load. Serialized as `Window`.
    #[serde(rename = "Window")]
    PlaneShared,
}

impl StageStorage {
    /// One dense block of shared memory in the source's order.
    pub const STRIDED: Self = StageStorage::Strided {
        order: StageOrder::AsSource,
    };

    /// This storage as a stage read by `instruction` and written by `delivery` lays its cells
    /// out: a tiled stage's rows arranged as both can use (`RowChunks::read_by`); any other
    /// storage as it is. What a host that names its instruction asks; one that names its Step
    /// asks the Step ([`ProductStep::stage_storage`](crate::ProductStep::stage_storage)).
    ///
    /// # Guarantees
    ///
    /// - The storage keeps its kind, its block and its order: only a tiled stage's row chunks
    ///   change.
    pub fn read_by(self, instruction: Instruction, delivery: Delivery) -> Self {
        match self {
            Self::Tiled {
                block,
                chunks,
                order,
            } => Self::Tiled {
                block,
                chunks: chunks.read_by(instruction, delivery),
                order,
            },
            other => other,
        }
    }

    /// The swizzle a TMA descriptor moving the whole of a stage over `space` in one box lands its
    /// rows with, so they lie as this storage keeps them: the stage's lines `vector_size` values of
    /// `elem_bits` bits, which a line holds whole bytes of (a packed operand's line is its words).
    /// What the launch builds the descriptor with, and what the stage fill holds the stage to.
    ///
    /// # Assumptions
    ///
    /// - A line of `vector_size` values of `elem_bits` bits is whole bytes.
    /// - The storage is in shared memory, and a tiled storage's block states an edge for every
    ///   dimension of `space` that divides its extent.
    ///
    /// # Errors
    ///
    /// Blocks splitting a row, which a box lands whole; blocks stacking a number of rows the
    /// swizzle's period of rows does not divide, whose rows a box keys off their place in the
    /// stage rather than in the block; padded rows, which a box lands dense, and swizzled rows
    /// longer than a descriptor swizzle spans.
    ///
    /// # Panics
    ///
    /// - Where a line is not whole bytes, or the storage is not shared memory, or a block lacks
    ///   or does not divide an edge.
    pub fn tma_swizzle(
        &self,
        space: &Space,
        vector_size: usize,
        elem_bits: usize,
    ) -> Result<TensorMapSwizzle, Decline> {
        assert!(
            (vector_size * elem_bits).is_multiple_of(8),
            "StageStorage::tma_swizzle: a line of {vector_size} {elem_bits}-bit values should be \
             whole bytes"
        );
        let form = StageForm::dense(
            space,
            vector_size,
            self.clone(),
            LineBytes(vector_size * elem_bits / 8),
        );
        let swizzle = form
            .rows
            .tma_swizzle()
            .map_err(|why| Decline::RowsNoDescriptorLands { why })?;
        let rank = space.rank();
        // A row spans the innermost dimension and those its swizzle merges beside it.
        let span = form.rows.swizzle().map_or(1, |swizzle| swizzle.span());
        let row = space.dimension_at(rank - 1);
        let period = match swizzle {
            TensorMapSwizzle::None => 1,
            _ => ChunkSwizzle::ROWS_PER_PERIOD,
        };
        for block in self.nesting(space).iter() {
            let rows = match rank > span {
                true => block.extent_at(rank - 1 - span),
                false => 1,
            };
            let splits_row = (rank - span..rank).any(|p| block.extent_at(p) < space.extent_at(p));
            if splits_row || !rows.is_multiple_of(period) {
                return Err(Decline::BoxSplitsStageBlocks {
                    dimension: row,
                    rows,
                    period,
                });
            }
        }
        Ok(swizzle)
    }

    /// Bytes a stage over `space` holds laid out this way, its lines `vector_size` values of
    /// `elem_bits` bits, which a line holds whole bytes of. [`Stages::smem`](crate::Stages::smem)
    /// allocates it once, rounding each plane's copy up to the stage alignment where planes own
    /// one each.
    ///
    /// # Assumptions
    ///
    /// - The storage is in shared memory, and a tiled storage's block states an edge for every
    ///   dimension of `space` that divides its extent.
    ///
    /// # Guarantees
    ///
    /// - The bytes cover every line of the block, and the padding after each of its rows, in the
    ///   order the storage names or, for a tiled stage in the source's order, in any order the
    ///   Lowering lays it down in for the fragments that read it
    ///   ([`ordered_for_fragments`](Self::ordered_for_fragments)).
    ///
    /// # Panics
    ///
    /// - Where the storage is not shared memory, or a block lacks or does not divide an edge.
    pub fn bytes(&self, space: &Space, vector_size: usize, elem_bits: usize) -> usize {
        let laid = self.laid_bytes(space, vector_size, elem_bits);
        let reordered = self
            .reordered()
            .into_iter()
            .filter(|_| elem_bits.is_multiple_of(8))
            .flat_map(|reorder| space.dimensions().map(reorder));
        reordered
            .map(|storage| storage.laid_bytes(space, 1, elem_bits))
            .fold(laid, usize::max)
    }
}

impl StageStorage {
    /// This storage as the Lowering lays a stage of an operand over `space` out for the fragments
    /// of a matrix instruction to read, the contraction summing it along `contracted` and its
    /// lines moved in by `delivery`: a tiled stage of unswizzled rows filled by the units' own
    /// loads and stores, in the order its fragments read at full width
    /// ([`StageOrder::for_fragments`]); any other as it is. A copy engine moves each source line
    /// whole and a swizzle permutes the chunks of the source's rows, so neither lays a stage in
    /// another order.
    pub(crate) fn ordered_for_fragments(
        self,
        space: &Space,
        contracted: Dimension,
        delivery: Delivery,
        mma: &MmaProperties,
    ) -> Self {
        match self {
            Self::Tiled {
                block,
                chunks,
                order: StageOrder::AsSource,
            } if chunks != RowChunks::Swizzled && delivery == Delivery::SyncPerUnit => {
                Self::Tiled {
                    block,
                    chunks,
                    order: StageOrder::for_fragments(space, contracted, mma),
                }
            }
            other => other,
        }
    }

    /// The order this storage lays a shared-memory stage down in.
    pub(crate) fn order(&self) -> StageOrder {
        match self {
            Self::Tiled { order, .. } | Self::Strided { order } => *order,
            Self::Lines { .. } => StageOrder::AsSource,
        }
    }

    /// Bytes a stage over `space` holds laid out in exactly this storage, `vector_size` values of
    /// `elem_bits` bits a line.
    fn laid_bytes(&self, space: &Space, vector_size: usize, elem_bits: usize) -> usize {
        let line = vector_size * elem_bits / 8;
        StageForm::dense_cells(space, vector_size, self, LineBytes(line)) * line
    }

    /// This storage laid with a Dimension moved innermost, where the Lowering may lay it so
    /// ([`ordered_for_fragments`](Self::ordered_for_fragments)): a tiled stage of unswizzled rows
    /// in the source's order.
    fn reordered(&self) -> Option<impl Fn(Dimension) -> StageStorage + '_> {
        match self {
            Self::Tiled {
                block,
                chunks,
                order: StageOrder::AsSource,
            } if *chunks != RowChunks::Swizzled => Some(move |dimension| Self::Tiled {
                block: block.clone(),
                chunks: *chunks,
                order: StageOrder::AlongRun(dimension),
            }),
            _ => None,
        }
    }

    /// The storage-tiling block a stage over `space` nests in its grid, `None` where it is one
    /// row-major block.
    pub(crate) fn nesting(&self, space: &Space) -> Option<Space> {
        match self {
            StageStorage::Lines { .. } => {
                panic!(
                    "StageStorage::Lines: the plane's units should not be laid out as shared memory"
                )
            }
            StageStorage::Tiled { block, .. } => block_space(block, space),
            StageStorage::Strided { .. } => None,
        }
    }
}

/// The block of edges `block` over `space`, `None` where it covers the whole of `space`.
fn block_space(block: &[(Dimension, usize)], space: &Space) -> Option<Space> {
    let nested = Space::new(
        &space
            .dimensions()
            .map(|dimension| {
                let edge = block
                    .iter()
                    .find(|&&(a, _)| a == dimension)
                    .unwrap_or_else(|| {
                        panic!(
                            "StageStorage::Tiled: the block should state an edge for {dimension:?}"
                        )
                    })
                    .1;
                (dimension, edge)
            })
            .collect::<Vec<_>>(),
    );
    (&nested != space).then_some(nested)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

    /// A target whose fragment load reads a run across its memory's rows one element at a time.
    fn element_wise() -> MmaProperties {
        MmaProperties {
            strided_run_read: StridedRunRead::ElementWise,
            ..MmaProperties::default()
        }
    }

    fn tiled(chunks: RowChunks) -> StageStorage {
        StageStorage::Tiled {
            block: vec![(M, 16), (N, 16), (K, 16)],
            chunks,
            order: StageOrder::AsSource,
        }
    }

    fn ordered(space: &Space, delivery: Delivery, mma: &MmaProperties) -> StageOrder {
        tiled(RowChunks::Padded)
            .ordered_for_fragments(space, K, delivery, mma)
            .order()
    }

    #[test]
    fn row_major_rhs_is_staged_along_k_where_runs_read_element_wise() {
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        let order = ordered(&rhs, Delivery::SyncPerUnit, &element_wise());
        assert_eq!(order, StageOrder::AlongRun(K));
    }

    /// An operand already laid `k` innermost, a row-major lhs, keeps its source's order.
    #[test]
    fn a_row_major_lhs_stays_as_source() {
        let lhs = Space::new(&[(M, 16), (K, 16)]);
        let order = ordered(&lhs, Delivery::SyncPerUnit, &element_wise());
        assert_eq!(order, StageOrder::AsSource);
    }

    #[test]
    fn stage_stays_as_source_where_runs_read_whole() {
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        let order = ordered(&rhs, Delivery::SyncPerUnit, &MmaProperties::default());
        assert_eq!(order, StageOrder::AsSource);
    }

    /// A copy engine moves each source line whole, so it lays no stage in another order.
    #[test]
    fn stage_stays_as_source_under_a_copy_engine() {
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        for delivery in [Delivery::AsyncPerUnit, Delivery::Tma] {
            let order = ordered(&rhs, delivery, &element_wise());
            assert_eq!(order, StageOrder::AsSource, "{delivery:?}");
        }
    }

    /// A swizzle permutes the chunks of the source's rows, so a swizzled stage keeps them.
    #[test]
    fn a_swizzled_stage_stays_as_source() {
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        let storage = tiled(RowChunks::Swizzled).ordered_for_fragments(
            &rhs,
            K,
            Delivery::SyncPerUnit,
            &element_wise(),
        );
        assert_eq!(storage.order(), StageOrder::AsSource);
    }

    /// A padded stage holding more cells along `n` than along `k` pads more rows laid along `k`:
    /// the bytes asked for a stage in the source's order cover the order the Lowering may lay.
    #[test]
    fn stage_bytes_cover_the_physical_order() {
        let rhs = Space::new(&[(K, 16), (N, 32)]);
        let storage = StageStorage::Tiled {
            block: vec![(K, 16), (N, 32)],
            chunks: RowChunks::Padded,
            order: StageOrder::AsSource,
        };
        let along_k = StageStorage::Tiled {
            block: vec![(K, 16), (N, 32)],
            chunks: RowChunks::Padded,
            order: StageOrder::AlongRun(K),
        };
        // 32 rows of 16 `f16` and one 16-byte chunk of padding, against 16 rows of 32 and one.
        assert_eq!(along_k.bytes(&rhs, 1, 16), 32 * (32 + 16));
        assert_eq!(storage.laid_bytes(&rhs, 8, 16), 16 * (64 + 16));
        assert_eq!(storage.bytes(&rhs, 8, 16), 32 * (32 + 16));
    }
}
