//! Where a stage lives and how it lays its cells out ([`StageStorage`]).

use crate::launch::Delivery;
use crate::{ChunkSwizzle, Decline, Dimension, Instruction, LineBytes, Space, StageForm};
use cubecl::ir::MmaProperties;
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
}

impl StageOrder {
    /// The order the fragments of `mma`'s target read the stage `stage` describes in at full
    /// width.
    ///
    /// # Guarantees
    ///
    /// - The source's order for a stage its fill does not copy as it lies
    ///   ([`StageSource::copied_as_it_lies`]), or that any delivery but the units' own loads and
    ///   stores ([`Delivery::SyncPerUnit`]) fills: a copy engine moves each source line whole.
    /// - The source's order on every target.
    pub(crate) fn for_fragments(stage: &FactorStage, _mma: &MmaProperties) -> Self {
        if stage.delivery != Delivery::SyncPerUnit || !stage.source.copied_as_it_lies() {
            return StageOrder::AsSource;
        }
        StageOrder::AsSource
    }
}

/// The facts of one stage of an operand that the order its fragments read it in is decided by
/// ([`StageStorage::ordered_for_fragments`]), alike where the Lowering lays the stage out and
/// where a host sizes it ([`StageStorage::bytes_read_by_fragments`]).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FactorStage {
    /// The Space the stage covers.
    pub space: Space,
    /// The Dimension a contraction sums the operand along, as the caller staging it states it:
    /// `None` for an operand it states none for, such as attention's keys and values.
    pub contracted: Option<Dimension>,
    /// The bits of one of the stage's values.
    pub elem_bits: usize,
    /// The values one line of the operand's source holds.
    pub vector_size: usize,
    /// Who moves the operand's bytes into the stage.
    pub delivery: Delivery,
    /// What the stage's fill does with the operand's values.
    pub source: StageSource,
}

/// What a stage's fill does with its operand's values: the facts that decide whether it copies
/// them as they lie ([`copied_as_it_lies`](Self::copied_as_it_lies)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StageSource {
    /// The stage holds the values the operand serves, neither decoded by its fill nor kept as
    /// the words they are stored in.
    pub served: bool,
    /// The operand's values are plain: not packed, not scaled and indexing no table.
    pub plain: bool,
    /// The stage's lines are as wide as the operand's own.
    pub own_width: bool,
    /// The operand is a window of memory, neither read through a tensor map nor computed.
    pub memory: bool,
    /// The operand's projection maps its Space directly onto its buffer, gathering nothing.
    pub direct: bool,
}

impl StageSource {
    /// Whether the stage's fill copies the operand's values as they lie: every fact holds.
    pub fn copied_as_it_lies(&self) -> bool {
        self.served && self.plain && self.own_width && self.memory && self.direct
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
    /// - The bytes cover every line of the block in its physical order, and the padding after
    ///   each of its rows.
    ///
    /// # Panics
    ///
    /// - Where the storage is not shared memory, or a block lacks or does not divide an edge.
    pub fn bytes(&self, space: &Space, vector_size: usize, elem_bits: usize) -> usize {
        let line = vector_size * elem_bits / 8;
        StageForm::dense_cells(space, vector_size, self, LineBytes(line)) * line
    }

    /// This storage as the Lowering lays out the stage `stage` describes, for the fragments of
    /// `mma`'s target to read: a tiled stage in the order they read it in
    /// ([`StageOrder::for_fragments`]), any other as it is.
    ///
    /// # Guarantees
    ///
    /// - The storage keeps its kind, its block and its row chunks.
    pub fn ordered_for_fragments(&self, stage: &FactorStage, mma: &MmaProperties) -> Self {
        match self {
            Self::Tiled { block, chunks, .. } => Self::Tiled {
                block: block.clone(),
                chunks: *chunks,
                order: StageOrder::for_fragments(stage, mma),
            },
            other => other.clone(),
        }
    }

    /// Bytes the stage `stage` describes holds where the fragments of `mma`'s target read it:
    /// [`bytes`](Self::bytes) of the storage the Lowering lays it out in
    /// ([`ordered_for_fragments`](Self::ordered_for_fragments)). What a host sizes a stage the
    /// Lowering builds by.
    ///
    /// # Assumptions
    ///
    /// - As [`bytes`](Self::bytes) over `stage.space`.
    /// - `stage` is the one the Lowering builds for the stage: one region's child Space of the
    ///   operand, the source's line width, the staged element's bits and what the fill does with
    ///   the operand's values.
    ///
    /// # Guarantees
    ///
    /// - The bytes are those the Lowering lays the stage out in, before it aligns the stage and
    ///   gives each plane a copy of its own.
    ///
    /// # Panics
    ///
    /// - As [`bytes`](Self::bytes).
    pub fn bytes_read_by_fragments(&self, stage: &FactorStage, mma: &MmaProperties) -> usize {
        self.ordered_for_fragments(stage, mma).bytes(
            &stage.space,
            stage.vector_size,
            stage.elem_bits,
        )
    }
}

impl StageStorage {
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

    /// A fill that copies its operand's values as they lie.
    const AS_IT_LIES: StageSource = StageSource {
        served: true,
        plain: true,
        own_width: true,
        memory: true,
        direct: true,
    };

    /// A row-major rhs `[k, n]` of `f16` in 8-wide lines copied as it lies, stated `contracted`
    /// and moved in by `delivery`.
    fn rhs(contracted: Option<Dimension>, delivery: Delivery) -> FactorStage {
        FactorStage {
            space: Space::new(&[(K, 16), (N, 32)]),
            contracted,
            elem_bits: 16,
            vector_size: 8,
            delivery,
            source: AS_IT_LIES,
        }
    }

    /// Every storage a host states for a factor's stage.
    fn storages() -> [StageStorage; 4] {
        let tiled = |chunks| StageStorage::Tiled {
            block: vec![(M, 16), (N, 16), (K, 16)],
            chunks,
            order: StageOrder::AsSource,
        };
        [
            tiled(RowChunks::InOrder),
            tiled(RowChunks::Swizzled),
            tiled(RowChunks::Padded),
            StageStorage::STRIDED,
        ]
    }

    /// The deliveries a host fills a factor's stage by, the units' own and a copy engine's.
    const DELIVERIES: [Delivery; 2] = [Delivery::SyncPerUnit, Delivery::AsyncPerUnit];

    /// On a default target, a stage stating `contracted` is laid out as stated.
    fn assert_laid_out_as_stated(contracted: Option<Dimension>) {
        let mma = MmaProperties::default();
        for delivery in DELIVERIES {
            let stage = rhs(contracted, delivery);
            for storage in storages() {
                let laid = storage.ordered_for_fragments(&stage, &mma);
                assert_eq!(laid, storage, "{delivery:?}");
            }
        }
    }

    /// On a default target, a stage stating `contracted` is sized as stated.
    fn assert_sized_as_stated(contracted: Option<Dimension>) {
        let mma = MmaProperties::default();
        for delivery in DELIVERIES {
            let stage = rhs(contracted, delivery);
            for storage in storages() {
                let bytes = storage.bytes(&stage.space, stage.vector_size, stage.elem_bits);
                let sized = storage.bytes_read_by_fragments(&stage, &mma);
                assert_eq!(sized, bytes, "{storage:?} {delivery:?}");
            }
        }
    }

    /// A matmul's factor, its contraction stated along `k`, is laid out as it states.
    #[test]
    fn a_factor_stating_its_contraction_is_laid_out_as_stated() {
        assert_laid_out_as_stated(Some(K));
    }

    /// A matmul's factor, its contraction stated along `k`, is sized as it states.
    #[test]
    fn a_factor_stating_its_contraction_is_sized_as_stated() {
        assert_sized_as_stated(Some(K));
    }

    /// An attention's keys and values, stating no contraction, are laid out as they state.
    #[test]
    fn a_stage_stating_no_contraction_is_laid_out_as_stated() {
        assert_laid_out_as_stated(None);
    }

    /// An attention's keys and values, stating no contraction, are sized as they state.
    #[test]
    fn a_stage_stating_no_contraction_is_sized_as_stated() {
        assert_sized_as_stated(None);
    }

    #[test]
    fn a_fill_copies_its_operand_as_it_lies_where_every_fact_holds() {
        assert!(AS_IT_LIES.copied_as_it_lies());
    }

    #[test]
    fn a_fill_copies_no_operand_as_it_lies_where_one_fact_fails() {
        let failing = [
            StageSource {
                served: false,
                ..AS_IT_LIES
            },
            StageSource {
                plain: false,
                ..AS_IT_LIES
            },
            StageSource {
                own_width: false,
                ..AS_IT_LIES
            },
            StageSource {
                memory: false,
                ..AS_IT_LIES
            },
            StageSource {
                direct: false,
                ..AS_IT_LIES
            },
        ];
        for source in failing {
            assert!(!source.copied_as_it_lies(), "{source:?}");
        }
    }
}
