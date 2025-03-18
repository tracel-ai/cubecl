use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// An approximation of the plane dimension.
pub const PLANE_DIM_APPROX: usize = 16;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[allow(missing_docs)]
pub struct CubeDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl CubeDim {
    /// Create a new cube dim with x = y = z = 1.
    pub const fn new_single() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new cube dim with the given x, y and z.
    /// This is equivalent to the [new](CubeDim::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total numbers of units per cube
    pub const fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for CubeDim {
    fn default() -> Self {
        Self {
            x: PLANE_DIM_APPROX as u32,
            y: PLANE_DIM_APPROX as u32,
            z: 1,
        }
    }
}

/// The kind of execution to be performed.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Unchecked kernels are unsafe.
    Unchecked,
}

/// Format of [`TensorMap`]
#[derive(Hash, PartialEq, Eq, Clone, Debug, Serialize, Deserialize)]
pub enum TensorMapFormat {
    /// Simple tiling
    Tiled {
        /// Tile size
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
