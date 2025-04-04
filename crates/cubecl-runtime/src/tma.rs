use alloc::vec::Vec;
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
use serde::{Deserialize, Serialize};

/// Format of [`TensorMap`]
#[cfg_attr(
    any(target_os = "windows", target_os = "linux", target_os = "macos"),
    derive(Serialize, Deserialize)
)]
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum TensorMapFormat {
    /// Simple tiling
    Tiled {
        /// Tile size that's loaded from memory in each copy operation. Must have `rank` elements.
        /// In matmul, for example, this might be `batch x m x k`, or whatever the stage size is.
        /// If a dimension isn't present in the tile, it should just be set to `1`.
        ///
        /// For CUDA, this must be a power of two and `<= 256` on each dimension.
        tile_size: Vec<u32>,
    },
    /// Im2col indexing. Loads a "column" (not the same column as im2col) of pixels into shared
    /// memory, with a certain offset (kernel position). The corners are the bounds to load pixels
    /// from *at offset 0*, so the top left corner of the kernel. The offset is added to the
    /// corner offsets, so a `(-1, -1)` corner will stop the bounding box at `(1, 1)` for kernel
    /// offset `(2, 2)`.
    Im2col {
        /// Pixel box lower corner. This is the logical upper left corner in the input tensor,
        /// when offset is 0. The length of this value should equal the *spatial* dimensions of
        /// the input tensor (i.e. `h, w` for an NHWC tensor). Should normally be set to `-padding`.
        pixel_box_lower_corner: Vec<i32>,
        /// Pixel box top corner. This is the logical lower right corner in the input tensor,
        /// when offset is 0. The length of this value should equal the *spatial* dimensions of
        /// the input tensor (i.e. `h, w` for an NHWC tensor). Should normally be set to
        /// `padding - kernel_size - 1` (where `kernel_size` accounts for dilation). This is not
        /// equal to padding, it's equal to the bounding box for the *top left corner of the kernel*.
        pixel_box_upper_corner: Vec<i32>,
        /// Channels to load per pixel, should be a multiple or divisor of the matmul tile size.
        /// This is not the total number of channels in the tensor, but only the number loaded in
        /// each load. Must be <= 256 and aligned to 16 bytes.
        channels_per_pixel: u32,
        /// Pixels per column, equivalent to the `m`/`n` dimension of each tile in the matrix
        /// multiplication. i.e. `NHW` for a 4D tensor.
        /// Must be <= 256 and aligned to 16 bytes
        pixels_per_column: u32,
    },
    /// Wide im2col
    Im2colWide {
        /// Pixel box lower corner width. TODO: How does this work?
        pixel_box_lower_corner_width: i32,
        /// Pixel box upper corner width. TODO: How does this work?
        pixel_box_upper_corner_width: i32,
        /// Channels per pixel
        channels_per_pixel: u32,
        /// Pixels per column
        pixels_per_column: u32,
    },
}

/// Interleave setting for [`TensorMap`]
#[cfg_attr(
    any(target_os = "windows", target_os = "linux", target_os = "macos"),
    derive(Serialize, Deserialize)
)]
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum TensorMapInterleave {
    /// No interleaving
    #[default]
    None,
    /// Interleaved with 16 bytes chunks in the last dim.
    /// i.e. NC/8HWC8 with f16
    B16,
    /// Interleaved with 32 bytes chunks in the last dim.
    /// i.e. NC/16HWC16 with f16
    B32,
}

/// Data are organized in a specific order in global memory; however, this may not match the order
/// in which the application accesses data in shared memory. This difference in data organization
/// may cause bank conflicts when shared memory is accessed. In order to avoid this problem, data
/// can be loaded to shared memory with shuffling across shared memory banks. When interleave is
/// [`TensorMapInterleave::B32`], swizzle must be [`TensorMapSwizzle::B32`].
/// Other interleave modes can have any swizzling pattern.
#[cfg_attr(
    any(target_os = "windows", target_os = "linux", target_os = "macos"),
    derive(Serialize, Deserialize)
)]
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum TensorMapSwizzle {
    /// No swizzling
    #[default]
    None,
    /// Swizzle 16B chunks within 32B span
    B32,
    /// Swizzle 16B chunks within 64B span
    B64,
    /// Swizzle 16B chunks within 128B span
    B128,
}

/// Additional prefetching to perform during load
/// Specifies L2 fetch size which indicates the byte granularity at which L2 requests are filled from DRAM
#[cfg_attr(
    any(target_os = "windows", target_os = "linux", target_os = "macos"),
    derive(Serialize, Deserialize)
)]
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum TensorMapPrefetch {
    /// No extra prefetch
    #[default]
    None,
    /// Prefetch 64 bytes
    B64,
    /// Prefetch 128 bytes
    B128,
    /// Prefetch 256 bytes
    B256,
}

/// What value to use when filling out of bounds values
#[cfg_attr(
    any(target_os = "windows", target_os = "linux", target_os = "macos"),
    derive(Serialize, Deserialize)
)]
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum OobFill {
    /// Fill zeroes
    #[default]
    Zero,
    /// Fill NaN
    NaN,
}
