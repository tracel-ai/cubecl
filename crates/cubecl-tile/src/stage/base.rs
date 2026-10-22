//! Where a stage lives and how it lays its cells out ([`StageStorage`]).

use crate::{Axis, ChunkSwizzle, LineBytes, Refusal, Space, StageForm};
use cubecl::prelude::TensorMapSwizzle;

/// Where a stage lives and how it lays its cells out, stated at
/// [`Stages::smem`](crate::Stages::smem).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    /// Shared memory grouped into `block`-sized tiles, one edge per axis of the operation.
    Tiled {
        block: Vec<(Axis, usize)>,
        chunks: RowChunks,
    },
    Strided,
    /// The plane's units, one line per unit.
    Lines {
        read: UnitRead,
    },
}

/// How a [`Tiled`](StageStorage::Tiled) block lays each row down, in 16-byte chunks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum RowChunks {
    /// Each row one contiguous run, in order; readable by `cmma` and TMA.
    InOrder,
    /// Each row's chunks XOR-permuted by a row key; only per-unit-addressed readers can read it.
    Swizzled,
    /// Each row in order followed by one unused chunk, so rows start on distinct banks.
    Padded,
}

impl RowChunks {
    /// Bytes one chunk holds: the widest read one unit issues.
    pub const CHUNK_BYTES: usize = 16;
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
    /// The swizzle a TMA descriptor moving the whole of a stage over `space` in one box lands its
    /// rows with, so they lie as this storage keeps them: the stage's lines `vector_size` values of
    /// `elem_bits` bits, which a line holds whole bytes of (a packed operand's line is its words).
    /// What the launch builds the descriptor with, and what the stage fill holds the stage to.
    ///
    /// # Errors
    ///
    /// Blocks splitting a row, which a box lands whole; blocks stacking a number of rows the
    /// swizzle's period of rows does not divide, whose rows a box keys off their place in the
    /// stage rather than in the block; padded rows, which a box lands dense, and swizzled rows
    /// longer than a descriptor swizzle spans.
    pub fn tma_swizzle(
        &self,
        space: &Space,
        vector_size: usize,
        elem_bits: usize,
    ) -> Result<TensorMapSwizzle, Refusal> {
        assert!(
            (vector_size * elem_bits).is_multiple_of(8),
            "StageStorage::tma_swizzle: a line of {vector_size} {elem_bits}-bit values is not whole \
             bytes"
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
            .map_err(|why| Refusal::RowsNoDescriptorLands { why })?;
        let rank = space.rank();
        let row = space.axis_at(rank - 1);
        let period = match swizzle {
            TensorMapSwizzle::None => 1,
            _ => ChunkSwizzle::ROWS_PER_PERIOD,
        };
        for block in self.nesting(space) {
            let rows = match rank {
                1 => 1,
                _ => block.extent_at(rank - 2),
            };
            if block.extent_at(rank - 1) < space.extent_at(rank - 1) || !rows.is_multiple_of(period)
            {
                return Err(Refusal::BoxSplitsStageBlocks {
                    axis: row,
                    rows,
                    period,
                });
            }
        }
        Ok(swizzle)
    }

    /// Bytes a stage over `space` holds laid out this way, its lines `vector_size` values of
    /// `elem_bits` bits, which a line holds whole bytes of: every line of its cells, and the
    /// padding after each row. What [`Stages::smem`](crate::Stages::smem) allocates for it.
    pub fn bytes(&self, space: &Space, vector_size: usize, elem_bits: usize) -> usize {
        let line = vector_size * elem_bits / 8;
        StageForm::dense(space, vector_size, self.clone(), LineBytes(line)).cells() * line
    }

    /// The storage-tiling nesting a stage over `space` gets, coarse to fine; empty is row-major.
    pub(crate) fn nesting(&self, space: &Space) -> Vec<Space> {
        match self {
            StageStorage::Lines { .. } => {
                panic!("StageStorage::Lines: the plane's units are not shared memory")
            }
            StageStorage::Tiled { block, .. } => {
                let nested = Space::new(
                    &space
                        .axes()
                        .map(|axis| {
                            let edge = block
                                .iter()
                                .find(|&&(a, _)| a == axis)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "StageStorage::Tiled: the block states no edge for {axis:?}"
                                    )
                                })
                                .1;
                            (axis, edge)
                        })
                        .collect::<Vec<_>>(),
                );
                if &nested == space {
                    return Vec::new();
                }
                vec![nested]
            }
            StageStorage::Strided => Vec::new(),
        }
    }
}
