//! Where a stage keeps each line of a block row ([`RowArrangement`]).

use cubecl::prelude::TensorMapSwizzle;

use crate::*;

/// Bytes one physical line of a buffer holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct LineBytes(pub(crate) usize);

/// Where a stage keeps each line of its innermost block's rows, resolved from [`RowChunks`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum RowArrangement {
    /// Each row one run, the next right after it: line `i` sits at offset `i`.
    InOrder,
    /// In order, each row's chunks permuted ([`ChunkSwizzle`]).
    Swizzled(ChunkSwizzle),
    /// Each row followed by `lines` unread lines, so each row starts further round the banks.
    Padded { lines: usize },
}

impl RowArrangement {
    /// Resolves `chunks` for a stage whose last two line extents are block rows and row lines.
    pub(crate) fn new(chunks: RowChunks, extents: &[usize], line: LineBytes) -> Self {
        let row_bytes = extents.last().copied().unwrap_or(0) * line.0;
        let one_chunk = row_bytes <= RowChunks::CHUNK_BYTES.max(line.0);
        match (chunks, one_chunk) {
            (RowChunks::InOrder, _) | (_, true) => Self::InOrder,
            (RowChunks::Swizzled, false) => match ChunkSwizzle::new(extents, line) {
                Some(swizzle) => Self::Swizzled(swizzle),
                None => Self::InOrder,
            },
            (RowChunks::Padded, false) => Self::Padded {
                lines: RowChunks::CHUNK_BYTES.div_ceil(line.0),
            },
        }
    }

    /// Whether a row is followed by padding, so line `i` is not at offset `i`.
    pub(crate) fn is_pitched(&self) -> bool {
        match self {
            Self::InOrder | Self::Swizzled(_) => false,
            Self::Padded { .. } => true,
        }
    }

    /// Whether each row is one run a stride after the previous one.
    pub(crate) fn rows_are_runs(&self) -> bool {
        match self {
            Self::InOrder | Self::Padded { .. } => true,
            Self::Swizzled(_) => false,
        }
    }

    /// The swizzle a TMA descriptor lands a box's rows with, so they sit as this stage keeps them:
    /// none for rows in order, the engine's own for a swizzled row of one span
    /// ([`ChunkSwizzle::tma`]).
    ///
    /// # Errors
    ///
    /// Padded rows, which a box lands dense, and a swizzled row longer than the engine swizzles.
    pub(crate) fn tma_swizzle(&self) -> Result<TensorMapSwizzle, &'static str> {
        match self {
            Self::InOrder => Ok(TensorMapSwizzle::None),
            Self::Swizzled(swizzle) => swizzle
                .tma()
                .ok_or("a swizzled row longer than 128 bytes, which no descriptor swizzle spans"),
            Self::Padded { .. } => Err("padded rows, which a descriptor lands dense"),
        }
    }

    /// Panics unless the rows are in order; `reader` and `why` go in the message.
    pub(crate) fn assert_in_order(&self, reader: &str, why: &str) {
        assert!(
            *self == Self::InOrder,
            "{reader}: {why}, so a {self:?} stage cannot be read by it; stage it RowChunks::InOrder"
        );
    }

    /// Panics if the rows are swizzled; `reader` goes in the message.
    pub(crate) fn assert_rows_are_runs(&self, reader: &str) {
        assert!(
            self.rows_are_runs(),
            "{reader}: a swizzled stage keeps a row's chunks out of order, so a fragment API \
             reading rows off a pointer and a stride (cmma) cannot read it; stage it \
             RowChunks::InOrder or RowChunks::Padded"
        );
    }

    /// Lines of padding after each row.
    pub(crate) fn padding(&self) -> usize {
        match self {
            Self::InOrder | Self::Swizzled(_) => 0,
            Self::Padded { lines } => *lines,
        }
    }

    /// The swizzle moving physical axis `axis`'s digits, if any.
    pub(crate) fn swizzle_along(&self, axis: usize) -> Option<ChunkSwizzle> {
        match self {
            Self::Swizzled(swizzle) if swizzle.line_axis() == axis => Some(*swizzle),
            Self::InOrder | Self::Swizzled(_) | Self::Padded { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swizzled() -> RowArrangement {
        RowArrangement::new(RowChunks::Swizzled, &[16, 4], LineBytes(16))
    }

    fn padded() -> RowArrangement {
        RowArrangement::new(RowChunks::Padded, &[16, 4], LineBytes(16))
    }

    #[test]
    #[should_panic(expected = "stage it RowChunks::InOrder")]
    fn a_dense_reader_refuses_a_swizzled_stage() {
        RowArrangement::InOrder.assert_in_order("reader", "why");
        swizzled().assert_in_order("reader", "why");
    }

    #[test]
    #[should_panic(expected = "stage it RowChunks::InOrder")]
    fn a_dense_reader_refuses_a_padded_stage() {
        padded().assert_in_order("reader", "why");
    }

    #[test]
    #[should_panic(expected = "a swizzled stage keeps a row's chunks out of order")]
    fn a_fragment_api_refuses_a_swizzled_stage() {
        RowArrangement::InOrder.assert_rows_are_runs("cmma");
        padded().assert_rows_are_runs("cmma");
        swizzled().assert_rows_are_runs("cmma");
    }

    #[test]
    fn a_request_resolves_against_the_rows_it_lays_down() {
        let extents = [2, 16, 4];
        assert_eq!(
            RowArrangement::new(RowChunks::InOrder, &extents, LineBytes(16)),
            RowArrangement::InOrder
        );
        assert!(matches!(
            RowArrangement::new(RowChunks::Swizzled, &extents, LineBytes(16)),
            RowArrangement::Swizzled(_)
        ));
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &extents, LineBytes(16)),
            RowArrangement::Padded { lines: 1 }
        );
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &[16, 1], LineBytes(16)),
            RowArrangement::InOrder
        );
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &[16, 16], LineBytes(4)),
            RowArrangement::Padded { lines: 4 }
        );
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &[16, 4], LineBytes(32)),
            RowArrangement::Padded { lines: 1 }
        );
    }
}
