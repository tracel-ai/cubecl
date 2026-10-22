//! Where a [`RowChunks::Swizzled`] stage keeps each line of a block row ([`ChunkSwizzle`]).

use cubecl::prelude::*;

use crate::*;

/// The XOR chunk swizzle a [`RowChunks::Swizzled`] stage resolves to: a line's chunk is
/// `XORed` with a key read off its row, so rows read together start on distinct banks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ChunkSwizzle {
    /// The physical axis holding a block's rows.
    row_axis: usize,
    /// The physical axis holding a block row's lines, the innermost.
    line_axis: usize,
    /// Lines one chunk holds.
    lines_per_chunk: usize,
    /// Consecutive rows taking one key: the rows one bank period holds.
    rows_per_key: usize,
    /// How many keys the rows cycle through, a power of two.
    keys: usize,
    /// Chunks one block row holds.
    row_chunks: usize,
}

impl ChunkSwizzle {
    /// Bytes one pass over every bank covers: 32 banks of four bytes.
    const BANK_PERIOD_BYTES: usize = 128;

    /// Rows after which the keys repeat, whatever the row's length: a row of a bank period or
    /// less cycles through `BANK_PERIOD_BYTES / CHUNK_BYTES` keys a key per period, one longer
    /// a key a row.
    pub(crate) const ROWS_PER_PERIOD: usize = Self::BANK_PERIOD_BYTES / RowChunks::CHUNK_BYTES;

    /// The swizzle of a stage with line `extents` (last two: block rows, row lines), `None` where
    /// there is nothing to permute.
    pub(crate) fn new(extents: &[usize], line: LineBytes) -> Option<Self> {
        let LineBytes(line_bytes) = line;
        let rank = extents.len();
        assert!(
            rank >= 2,
            "ChunkSwizzle: a swizzled stage permutes the chunks of a row, and a rank-{rank} stage \
             has no rows"
        );
        assert!(
            line_bytes.is_power_of_two(),
            "ChunkSwizzle: a {line_bytes}-byte line does not tile a chunk"
        );
        let chunk_bytes = RowChunks::CHUNK_BYTES.max(line_bytes);
        let lines_per_chunk = chunk_bytes / line_bytes;
        let row_lines = extents[rank - 1];
        if row_lines * line_bytes <= chunk_bytes {
            return None;
        }
        assert!(
            row_lines.is_multiple_of(lines_per_chunk)
                && (row_lines / lines_per_chunk).is_power_of_two(),
            "ChunkSwizzle: a row of {row_lines} {line_bytes}-byte lines is not a power of two of \
             {chunk_bytes}-byte chunks, which an XOR permutes"
        );
        let chunks = row_lines / lines_per_chunk;
        let period_chunks = Self::BANK_PERIOD_BYTES / chunk_bytes;
        let keys = chunks.min(period_chunks);
        if keys < 2 {
            return None;
        }
        Some(Self {
            row_axis: rank - 2,
            line_axis: rank - 1,
            lines_per_chunk,
            rows_per_key: (period_chunks / chunks).max(1),
            keys,
            row_chunks: chunks,
        })
    }

    /// The swizzle a TMA descriptor applies to land a box's rows where this one keeps them, where
    /// one does: a row of one 32, 64 or 128-byte span. The engine XORs a row's chunk with the
    /// address bits above the bank period, which for such a row are the row's own bits this
    /// swizzle keys on, provided the stage starts on a span of the bank period's rows
    /// (the TMA stage alignment). A longer row has no descriptor
    /// swizzle: the engine's spans stop at 128 bytes.
    pub(crate) fn tma(&self) -> Option<TensorMapSwizzle> {
        match self.row_chunks * RowChunks::CHUNK_BYTES {
            32 => Some(TensorMapSwizzle::B32),
            64 => Some(TensorMapSwizzle::B64),
            128 => Some(TensorMapSwizzle::B128),
            _ => None,
        }
    }

    /// The physical axis holding a block's rows, whose digit keys the swizzle.
    pub(crate) fn row_axis(&self) -> usize {
        self.row_axis
    }

    /// The physical axis holding a block row's lines, whose digit the swizzle moves.
    pub(crate) fn line_axis(&self) -> usize {
        self.line_axis
    }

    /// Host twin of [`swizzled_line`].
    #[cfg(test)]
    fn line_of(&self, line: usize, row: usize) -> usize {
        line ^ ((row / self.rows_per_key) % self.keys * self.lines_per_chunk)
    }
}

/// Where line `line` of block row `row` is kept under `swizzle`; its own inverse.
#[cube]
pub(crate) fn swizzled_line(#[comptime] swizzle: ChunkSwizzle, line: u32, row: u32) -> u32 {
    let key = row
        .divided_by(comptime!(swizzle.rows_per_key as u32).runtime())
        .remainder(comptime!(swizzle.keys as u32).runtime());
    line ^ key.times(comptime!(swizzle.lines_per_chunk as u32).runtime())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte a line starts at, within its bank period.
    fn bank_offset(
        swizzle: &ChunkSwizzle,
        line_bytes: usize,
        row_lines: usize,
        row: usize,
    ) -> usize {
        let line = swizzle.line_of(0, row);
        (row * row_lines + line) * line_bytes % ChunkSwizzle::BANK_PERIOD_BYTES
    }

    /// Eight rows of a 64-byte block start on distinct 16-byte bank slots.
    #[test]
    fn eight_rows_of_a_64_byte_block_start_on_distinct_banks() {
        let swizzle = ChunkSwizzle::new(&[2, 16, 4], LineBytes(16)).unwrap();
        let mut slots: Vec<usize> = (0..8)
            .map(|row| bank_offset(&swizzle, 16, 4, row))
            .collect();
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), 8);
    }

    #[test]
    fn rows_short_and_long_start_on_distinct_banks() {
        for (row_lines, line_bytes) in [(2, 16), (16, 16), (8, 8), (16, 4)] {
            let swizzle = ChunkSwizzle::new(&[16, row_lines], LineBytes(line_bytes))
                .expect("a row of several chunks is swizzled");
            let rows = ChunkSwizzle::BANK_PERIOD_BYTES / RowChunks::CHUNK_BYTES;
            let mut slots: Vec<usize> = (0..rows)
                .map(|row| {
                    bank_offset(&swizzle, line_bytes, row_lines, row) / RowChunks::CHUNK_BYTES
                })
                .collect();
            slots.sort();
            slots.dedup();
            assert_eq!(slots.len(), rows, "{row_lines} lines of {line_bytes} bytes");
        }
    }

    #[test]
    fn a_row_is_permuted_within_itself_a_chunk_at_a_time() {
        let swizzle = ChunkSwizzle::new(&[16, 16], LineBytes(4)).unwrap();
        for row in 0..16 {
            let mut lines: Vec<usize> = (0..16).map(|line| swizzle.line_of(line, row)).collect();
            for line in 0..16 {
                assert_eq!(swizzle.line_of(line, row) % 4, line % 4);
                assert_eq!(swizzle.line_of(swizzle.line_of(line, row), row), line);
            }
            lines.sort();
            assert_eq!(lines, (0..16).collect::<Vec<_>>());
        }
    }

    /// Where the TMA engine lands chunk `chunk` of row `row` under its swizzle of a
    /// `row_bytes`-byte row: CUTLASS's `Swizzle<B, 4, 3>`, the 16-byte chunk bits `XORed` with the
    /// bits above the 128-byte bank period.
    fn tma_chunk(row_bytes: usize, row: usize, chunk: usize) -> usize {
        let bits = (row_bytes / RowChunks::CHUNK_BYTES).trailing_zeros();
        let address = row * row_bytes + chunk * RowChunks::CHUNK_BYTES;
        chunk ^ ((address >> 7) & ((1 << bits) - 1))
    }

    /// A row of one 32, 64 or 128-byte span is swizzled as the TMA engine lands it, so a
    /// descriptor fills the stage a fragment reads swizzled; a longer row has no descriptor
    /// swizzle.
    #[test]
    fn a_row_of_one_tma_span_is_swizzled_as_the_engine_lands_it() {
        for (row_lines, line_bytes, mode) in [
            (16, 2, Some(TensorMapSwizzle::B32)),
            (2, 16, Some(TensorMapSwizzle::B32)),
            (32, 2, Some(TensorMapSwizzle::B64)),
            (64, 2, Some(TensorMapSwizzle::B128)),
            (8, 16, Some(TensorMapSwizzle::B128)),
            (128, 2, None),
        ] {
            let swizzle = ChunkSwizzle::new(&[64, row_lines], LineBytes(line_bytes)).unwrap();
            assert_eq!(
                swizzle.tma(),
                mode,
                "{row_lines} lines of {line_bytes} bytes"
            );
            if mode.is_none() {
                continue;
            }
            let row_bytes = row_lines * line_bytes;
            let lines_per_chunk = RowChunks::CHUNK_BYTES / line_bytes;
            for row in 0..64 {
                for chunk in 0..row_bytes / RowChunks::CHUNK_BYTES {
                    let line = swizzle.line_of(chunk * lines_per_chunk, row);
                    assert_eq!(
                        line / lines_per_chunk,
                        tma_chunk(row_bytes, row, chunk),
                        "row {row}, chunk {chunk} of a {row_bytes}-byte row"
                    );
                }
            }
        }
    }

    /// A stage of K-major rows one TMA span long lies as a warpgroup MMA reads a tile under the
    /// same swizzle, so a descriptor over the stage reads what the TMA engine landed.
    #[test]
    fn a_tma_span_row_is_the_warpgroup_mma_tile() {
        use cubecl::wgmma::{Major, Swizzle, WgmmaTileLayout};

        const F16_BYTES: usize = 2;
        for (row_bytes, swizzle) in [(32, Swizzle::B32), (64, Swizzle::B64), (128, Swizzle::B128)] {
            let row_lines = row_bytes / F16_BYTES;
            let stage = ChunkSwizzle::new(&[64, row_lines], LineBytes(F16_BYTES)).unwrap();
            let tile = WgmmaTileLayout {
                major: Major::K,
                swizzle,
                rows: 64,
                k: row_lines,
            };
            assert_eq!(tile.validate(F16_BYTES), Ok(()), "{tile:?}");
            for row in 0..64 {
                for k in 0..row_lines {
                    assert_eq!(
                        row * row_lines + stage.line_of(k, row),
                        tile.offset(row, k, F16_BYTES),
                        "{row_bytes}-byte rows, row {row}, k {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_row_of_one_chunk_is_left_in_order() {
        assert!(ChunkSwizzle::new(&[16, 1], LineBytes(16)).is_none());
        assert!(ChunkSwizzle::new(&[16, 2], LineBytes(8)).is_none());
        assert!(ChunkSwizzle::new(&[16, 4], LineBytes(128)).is_none());
    }
}
