use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Format of [`TensorMap`]
#[derive(Hash, PartialEq, Eq, Clone, Debug, Serialize, Deserialize)]
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
    /// Im2col indexing
    Im2col {
        /// Pixel box lower corner. TODO: How does this work?
        pixel_box_lower_corner: Vec<i32>,
        /// Pixel box upper corner. TODO: How does this work?
        pixel_box_upper_corner: Vec<i32>,
        /// Channels per pixel
        channels_per_pixel: u32,
        /// Pixels per column, aka kernel size
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
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy, Serialize, Deserialize)]
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
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy, Serialize, Deserialize)]
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
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy, Serialize, Deserialize)]
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
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy, Serialize, Deserialize)]
pub enum OobFill {
    /// Fill zeroes
    #[default]
    Zero,
    /// Fill NaN
    NaN,
}
