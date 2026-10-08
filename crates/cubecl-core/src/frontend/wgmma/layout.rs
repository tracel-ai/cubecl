//! The shared memory layouts a warpgroup MMA reads its operand tiles in.

use crate::{self as cubecl, prelude::*};
use cubecl_runtime::tma::TensorMapSwizzle;

pub use cubecl_ir::dialect::matrix::{WgmmaMajor as Major, WgmmaSwizzle as Swizzle};

/// Bytes of K every warpgroup MMA reads.
pub(crate) const K_BYTES: usize = 32;
/// Every line of a swizzle pattern is 16-byte chunks, and the pattern repeats every 8 lines.
const CHUNK_BYTES: usize = 16;
const PATTERN_LINES: usize = 8;

/// A `rows x k` operand tile in shared memory, laid out the way a warpgroup MMA reads it. `rows`
/// runs along M for `A` and along N for `B`, and `major` names the contiguous dimension.
///
/// The tile is cut along its contiguous dimension into panels as wide as the swizzle, 16 bytes
/// without one. A panel holds a line of the contiguous dimension for each index of the other,
/// one after the other, with the 16-byte chunks of each line permuted by the swizzle. The panels
/// follow each other. That is what a TMA load writes with [`Self::tensor_map_swizzle`] and a box
/// one panel wide.
///
/// A tile starts at a multiple of [`Self::alignment`] in shared memory, since the swizzle
/// permutes by address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WgmmaTileLayout {
    pub major: Major,
    pub swizzle: Swizzle,
    pub rows: usize,
    pub k: usize,
}

/// A layout no warpgroup MMA can read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgmmaLayoutError {
    /// The lines of the contiguous dimension don't fill whole panels.
    PartialPanel {
        line_bytes: usize,
        panel_bytes: usize,
    },
    /// The swizzle pattern repeats every 8 lines, and the tile has a partial one.
    PartialPattern { lines: usize },
    /// K isn't a whole number of MMA steps.
    PartialStep { k: usize, k_step: usize },
    /// Only 16-bit elements may be MN-major.
    MnMajor { elem_size: usize },
}

impl core::fmt::Display for WgmmaLayoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PartialPanel {
                line_bytes,
                panel_bytes,
            } => write!(
                f,
                "a line of {line_bytes} bytes doesn't fill whole {panel_bytes}-byte panels"
            ),
            Self::PartialPattern { lines } => write!(
                f,
                "{lines} lines isn't a multiple of the {PATTERN_LINES} the swizzle repeats over"
            ),
            Self::PartialStep { k, k_step } => {
                write!(f, "a K of {k} isn't a multiple of the MMA's {k_step}")
            }
            Self::MnMajor { elem_size } => write!(
                f,
                "only 16-bit elements may be MN-major, and these are {} bits",
                elem_size * 8
            ),
        }
    }
}

impl WgmmaTileLayout {
    /// The elements of K one warpgroup MMA reads.
    pub fn k_step(elem_size: usize) -> usize {
        K_BYTES / elem_size
    }

    /// The alignment, in bytes, the tile must start at.
    pub fn alignment(&self) -> usize {
        self.swizzle.alignment()
    }

    /// The bytes the tile spans.
    pub fn size_bytes(&self, elem_size: usize) -> usize {
        self.rows * self.k * elem_size
    }

    /// The swizzle a TMA load of the tile uses.
    pub fn tensor_map_swizzle(&self) -> TensorMapSwizzle {
        match self.swizzle {
            Swizzle::None => TensorMapSwizzle::None,
            Swizzle::B32 => TensorMapSwizzle::B32,
            Swizzle::B64 => TensorMapSwizzle::B64,
            Swizzle::B128 => TensorMapSwizzle::B128,
        }
    }

    /// The elements of the contiguous dimension a panel holds, the inner dimension of the box a
    /// TMA load of the tile uses.
    pub fn panel_elems(&self, elem_size: usize) -> usize {
        self.panel_bytes() / elem_size
    }

    /// Checks that a warpgroup MMA can read the tile.
    pub fn validate(&self, elem_size: usize) -> Result<(), WgmmaLayoutError> {
        if self.major == Major::MN && elem_size != 2 {
            return Err(WgmmaLayoutError::MnMajor { elem_size });
        }
        let line_bytes = self.line_elems() * elem_size;
        let panel_bytes = self.panel_bytes();
        if !line_bytes.is_multiple_of(panel_bytes) {
            return Err(WgmmaLayoutError::PartialPanel {
                line_bytes,
                panel_bytes,
            });
        }
        let lines = self.lines();
        if !lines.is_multiple_of(PATTERN_LINES) {
            return Err(WgmmaLayoutError::PartialPattern { lines });
        }
        let k_step = Self::k_step(elem_size);
        if !self.k.is_multiple_of(k_step) {
            return Err(WgmmaLayoutError::PartialStep { k: self.k, k_step });
        }
        Ok(())
    }

    /// The descriptor's leading dimension byte offset: between panels, or for an MN-major tile
    /// without swizzle, between groups of 8 lines. A K-major swizzled tile has a single panel
    /// per MMA, and the field is ignored.
    pub fn leading_byte_offset(&self) -> usize {
        match (self.major, self.swizzle) {
            (Major::K, Swizzle::None) => self.panel_stride(),
            (Major::K, _) => CHUNK_BYTES,
            (Major::MN, Swizzle::None) => self.pattern_stride(),
            (Major::MN, _) => self.panel_stride(),
        }
    }

    /// The descriptor's stride dimension byte offset: between groups of 8 lines, or for an
    /// MN-major tile without swizzle, between panels.
    pub fn stride_byte_offset(&self) -> usize {
        match (self.major, self.swizzle) {
            (Major::MN, Swizzle::None) => self.panel_stride(),
            _ => self.pattern_stride(),
        }
    }

    /// The bytes from the start of the tile to element `(row, k)`, before the swizzle. A
    /// descriptor of the tile moved this far reads the sub-tile starting there.
    pub fn byte_offset(&self, row: usize, k: usize, elem_size: usize) -> usize {
        let (along, line) = match self.major {
            Major::K => (k, row),
            Major::MN => (row, k),
        };
        let along = along * elem_size;
        let panel_bytes = self.panel_bytes();
        (along / panel_bytes) * self.panel_stride() + line * panel_bytes + along % panel_bytes
    }

    /// The index of element `(row, k)` in the tile, swizzle included.
    pub fn offset(&self, row: usize, k: usize, elem_size: usize) -> usize {
        let bytes = self.byte_offset(row, k, elem_size);
        let pattern = (bytes / (CHUNK_BYTES * PATTERN_LINES)) & self.chunk_mask();
        (bytes ^ (pattern * CHUNK_BYTES)) / elem_size
    }

    /// Expand method of [`Self::byte_offset`].
    pub fn __expand_byte_offset_method(
        &self,
        scope: &Scope,
        row: NativeExpand<usize>,
        k: NativeExpand<usize>,
        elem_size: usize,
    ) -> NativeExpand<usize> {
        let (along, line) = match self.major {
            Major::K => (k, row),
            Major::MN => (row, k),
        };
        byte_offset::expand(scope, along, line, *self, elem_size)
    }

    /// Expand method of [`Self::offset`].
    pub fn __expand_offset_method(
        &self,
        scope: &Scope,
        row: NativeExpand<usize>,
        k: NativeExpand<usize>,
        elem_size: usize,
    ) -> NativeExpand<usize> {
        let bytes = self.__expand_byte_offset_method(scope, row, k, elem_size);
        swizzle::expand(scope, bytes, self.chunk_mask(), elem_size)
    }

    fn panel_bytes(&self) -> usize {
        match self.swizzle {
            Swizzle::None => CHUNK_BYTES,
            Swizzle::B32 => 32,
            Swizzle::B64 => 64,
            Swizzle::B128 => 128,
        }
    }

    /// The elements of the contiguous dimension.
    fn line_elems(&self) -> usize {
        match self.major {
            Major::K => self.k,
            Major::MN => self.rows,
        }
    }

    /// The lines of a panel, one per index of the dimension that isn't contiguous.
    fn lines(&self) -> usize {
        match self.major {
            Major::K => self.rows,
            Major::MN => self.k,
        }
    }

    fn panel_stride(&self) -> usize {
        self.lines() * self.panel_bytes()
    }

    fn pattern_stride(&self) -> usize {
        PATTERN_LINES * self.panel_bytes()
    }

    /// The bits of a chunk's index in its line that the swizzle permutes.
    fn chunk_mask(&self) -> usize {
        self.panel_bytes() / CHUNK_BYTES - 1
    }
}

/// [`WgmmaTileLayout::byte_offset`], of the element `along` the contiguous dimension in `line`.
#[cube]
fn byte_offset(
    along: usize,
    line: usize,
    #[comptime] layout: WgmmaTileLayout,
    #[comptime] elem_size: usize,
) -> usize {
    let panel_bytes = comptime![layout.panel_bytes()];
    let along = along * elem_size;
    (along / panel_bytes) * comptime![layout.panel_stride()]
        + line * panel_bytes
        + along % panel_bytes
}

/// The element at `bytes` once the swizzle permuted its chunk.
#[cube]
fn swizzle(bytes: usize, #[comptime] chunk_mask: usize, #[comptime] elem_size: usize) -> usize {
    let pattern = (bytes / comptime![CHUNK_BYTES * PATTERN_LINES]) & chunk_mask;
    (bytes ^ (pattern * CHUNK_BYTES)) / elem_size
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(major: Major, swizzle: Swizzle, rows: usize, k: usize) -> WgmmaTileLayout {
        WgmmaTileLayout {
            major,
            swizzle,
            rows,
            k,
        }
    }

    /// PTX ISA, "Matrix Descriptor Format", K-major without swizzle, `tf32`: `T = 4`, two groups
    /// of 8 rows and two core matrices along K. The exact layout
    /// `((8,2),(4,4)):((4,32),(1,64))` gives an LBO of 64 and an SBO of 32 elements.
    #[test]
    fn k_major_no_swizzle_matches_the_ptx_example() {
        let tile = layout(Major::K, Swizzle::None, 16, 8);
        assert_eq!(tile.leading_byte_offset(), 64 * 4);
        assert_eq!(tile.stride_byte_offset(), 32 * 4);
        for (row, k) in [(0, 0), (1, 0), (7, 3), (8, 0), (0, 4), (15, 7)] {
            let expected = (row % 8) * 4 + (row / 8) * 32 + (k % 4) + (k / 4) * 64;
            assert_eq!(tile.offset(row, k, 4), expected, "({row}, {k})");
        }
    }

    /// PTX ISA, K-major with 32-byte swizzle, `tf32`: the exact layout
    /// `((8,2),(4,2)):((8,64),(1,4))` gives an SBO of 64 elements, and the LBO is unused.
    #[test]
    fn k_major_32b_matches_the_ptx_example() {
        let tile = layout(Major::K, Swizzle::B32, 16, 8);
        assert_eq!(tile.stride_byte_offset(), 64 * 4);
        for (row, k) in [(0, 0), (1, 0), (7, 3), (8, 0), (0, 4), (15, 7)] {
            let unswizzled = (row % 8) * 8 + (row / 8) * 64 + k;
            assert_eq!(tile.byte_offset(row, k, 4), unswizzled * 4, "({row}, {k})");
        }
    }

    /// PTX ISA, MN-major with 32-byte swizzle, `bf16`: the exact layout
    /// `((8,2,2),(8,2)):((1,8,128),(16,256))`, panels of 8 lines of K, gives an LBO of 128 and
    /// an SBO of 256 elements. A tile of one 8-line group of K has the same panels.
    #[test]
    fn mn_major_32b_matches_the_ptx_example() {
        let tile = layout(Major::MN, Swizzle::B32, 32, 8);
        assert_eq!(tile.leading_byte_offset(), 128 * 2);
        assert_eq!(tile.stride_byte_offset(), 8 * 32);
        for (mn, k) in [(0, 0), (9, 0), (16, 3), (31, 7)] {
            let unswizzled = (mn % 8) + ((mn / 8) % 2) * 8 + (mn / 16) * 128 + k * 16;
            assert_eq!(tile.byte_offset(mn, k, 2), unswizzled * 2, "({mn}, {k})");
        }
    }

    /// PTX ISA, MN-major with 64-byte swizzle, `bf16`: the exact layout
    /// `((8,4,2),(8,2)):((1,8,256),(32,512))` gives an LBO of 256 elements between panels of 8
    /// lines.
    #[test]
    fn mn_major_64b_matches_the_ptx_example() {
        let tile = layout(Major::MN, Swizzle::B64, 64, 8);
        assert_eq!(tile.leading_byte_offset(), 256 * 2);
        assert_eq!(tile.stride_byte_offset(), 8 * 64);
        for (mn, k) in [(0, 0), (9, 1), (32, 0), (63, 7)] {
            let unswizzled = (mn % 32) + (mn / 32) * 256 + k * 32;
            assert_eq!(tile.byte_offset(mn, k, 2), unswizzled * 2, "({mn}, {k})");
        }
    }

    /// PTX ISA, MN-major without swizzle, `bf16`: groups of 8 elements along MN are 16 bytes
    /// apart within a group of 8 lines of K, `((8,1,2),(8,2))`.
    #[test]
    fn mn_major_no_swizzle_uses_core_matrices() {
        let tile = layout(Major::MN, Swizzle::None, 16, 16);
        assert_eq!(tile.leading_byte_offset(), 128);
        assert_eq!(tile.stride_byte_offset(), 16 * 16);
        assert_eq!(tile.offset(0, 1, 2), 8);
        assert_eq!(tile.offset(8, 0, 2), 128);
    }

    /// `Swizzle<3,4,3>`: the chunk index in a 128-byte line is `XOR`ed with the line's index in
    /// its group of 8.
    #[test]
    fn swizzle_128b_permutes_chunks_by_line() {
        let tile = layout(Major::K, Swizzle::B128, 64, 64);
        for line in 0..8 {
            for chunk in 0..8 {
                let k = chunk * 8;
                let expected = line * 64 + (chunk ^ line) * 8;
                assert_eq!(
                    tile.offset(line, k, 2),
                    expected,
                    "line {line}, chunk {chunk}"
                );
            }
        }
        // The next group of 8 lines repeats the pattern.
        assert_eq!(tile.offset(9, 0, 2), 9 * 64 + 8);
    }

    /// `Swizzle<2,4,3>` and `Swizzle<1,4,3>` XOR the chunk with bits 7 and up of the address,
    /// which a narrow line reaches every 2 or 4 lines.
    #[test]
    fn narrow_swizzles_permute_by_address() {
        let b64 = layout(Major::K, Swizzle::B64, 8, 32);
        assert_eq!(b64.offset(1, 0, 2), 32);
        assert_eq!(b64.offset(2, 0, 2), 64 + 8);
        assert_eq!(b64.offset(6, 8, 2), 6 * 32 + (1 ^ 3) * 8);

        let b32 = layout(Major::K, Swizzle::B32, 8, 16);
        assert_eq!(b32.offset(3, 0, 2), 3 * 16);
        assert_eq!(b32.offset(4, 0, 2), 4 * 16 + 8);
        assert_eq!(b32.offset(4, 8, 2), 4 * 16);
    }

    /// A K-major tile wider than its swizzle is panels side by side, each spanning every row.
    #[test]
    fn k_major_panels_follow_each_other() {
        let tile = layout(Major::K, Swizzle::B64, 64, 64);
        assert_eq!(tile.byte_offset(0, 32, 2), 64 * 64);
        assert_eq!(tile.byte_offset(1, 33, 2), 64 * 64 + 64 + 2);
        // The second MMA step starts 32 bytes into the first panel.
        assert_eq!(tile.byte_offset(0, 16, 2), 32);
    }

    #[test]
    fn validate_rejects_layouts_no_mma_reads() {
        let partial_panel = layout(Major::K, Swizzle::B128, 64, 32);
        assert_eq!(
            partial_panel.validate(2),
            Err(WgmmaLayoutError::PartialPanel {
                line_bytes: 64,
                panel_bytes: 128
            })
        );
        let mn_fp8 = layout(Major::MN, Swizzle::B128, 128, 64);
        assert_eq!(
            mn_fp8.validate(1),
            Err(WgmmaLayoutError::MnMajor { elem_size: 1 })
        );
        let partial_step = layout(Major::MN, Swizzle::B128, 64, 8);
        assert_eq!(
            partial_step.validate(2),
            Err(WgmmaLayoutError::PartialStep { k: 8, k_step: 16 })
        );
        assert_eq!(layout(Major::K, Swizzle::B128, 64, 64).validate(2), Ok(()));
    }
}
