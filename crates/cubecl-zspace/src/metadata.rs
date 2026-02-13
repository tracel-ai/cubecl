//! # Tensor Metadata Types
//!
//! This module provides fundamental structures for describing how tensor data is
//! organized in memory, including dimensions (shape) and memory mapping (layout).

use core::ops::Deref;
use sharded_slab::{Clear, Pool};
use smallvec::SmallVec;
use std::{ops::DerefMut, sync::OnceLock};

static POOL: OnceLock<Pool<Metadata>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MetadataHandle {
    key: usize,
}

impl MetadataHandle {
    pub fn metadata(&self) -> impl Deref<Target = Metadata> {
        let pool = POOL.get_or_init(|| Pool::new());
        pool.get(self.key).unwrap()
    }

    pub fn new() -> (Self, impl DerefMut<Target = Metadata>) {
        let pool = POOL.get_or_init(|| Pool::new());
        let entry = pool.create().unwrap();

        let id = entry.key();

        let this = Self { key: id };

        (this, entry)
    }
}

/// Metadata describing a tensor's structure and memory organization.
///
/// These structures use `SmallVec` for indices, making them cheap to clone
/// for tensors with a rank (number of dimensions) up to 8.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Metadata {
    /// The dimensions of the tensor.
    pub shape: Indices,
    /// The mapping of tensor coordinates to physical memory offsets.
    pub layout: Layout,
}

impl Clear for Metadata {
    fn clear(&mut self) {}
}

/// Defines how elements are mapped to the underlying data buffer.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Layout {
    /// Standard row-major layout where the last axis is contiguous.
    ///
    /// An N-dimensional index $(i_0, i_1, ..., i_n)$ maps to an offset via strides
    /// calculated from the shape, where the stride of the last dimension is 1.
    #[default]
    Contiguous,

    /// A layout where each dimension has a custom step size (stride) in memory.
    ///
    /// The offset is calculated as: $\sum (index_i \times stride_i)$.
    Strided {
        /// The memory step for each dimension.
        strides: Indices,
    },

    /// A 2D-tiled memory layout optimized for hardware cache or matrix operations.
    ///
    /// For a 4D tensor like `[batch, seq, channel, d_model]`, this allows
    /// the last two dimensions to be stored in smaller, contiguous "tiles".
    Tiled2D {
        /// Strides for the higher-level (non-tiled) dimensions.
        strides: Indices,
        /// Dimensions of a single tile: `(width, height)`.
        tile_size: (u16, u16),
        /// Total grid size of tiles: `(columns, rows)`.
        tile_count: (usize, usize),
        /// Memory order of elements within an individual tile.
        layout_inner: Tile2DLayout,
        /// Memory order of the tiles themselves within the buffer.
        layout_outer: Tile2DLayout,
    },
}

/// A specialized container for tensor dimensions or strides.
///
/// Uses a stack-allocated buffer for up to 8 dimensions to avoid heap allocations
/// in common deep learning workloads.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Indices {
    data: SmallVec<[usize; 8]>,
}

/// Memory ordering for 2D structures (Tiled or Matrix).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile2DLayout {
    /// Elements/Tiles are stored column-by-column.
    ColMajor,
    /// Elements/Tiles are stored row-by-row.
    RowMajor,
}

impl Indices {
    /// Creates a new `Indices` instance from a slice.
    pub fn new(indices: &[usize]) -> Self {
        Self {
            data: SmallVec::from_slice(indices),
        }
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.data.len()
    }
}

/// Allows `Indices` to be treated like a slice (e.g., `indices[0]` or `.iter()`).
impl Deref for Indices {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl From<&[usize]> for Indices {
    fn from(s: &[usize]) -> Self {
        Self::new(s)
    }
}
