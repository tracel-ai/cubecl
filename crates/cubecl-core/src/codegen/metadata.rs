//! Metadata helpers to easily get offsets etc.
//!
//! Conceptually, metadata is represented like this:
//! ```rust
//! struct Metadata<const NUM_BUFS: usize, const NUM_EXT: usize> {
//!     base: BaseMeta<NUM_BUFS>,
//!     extended: ExtendedMeta<NUM_EXT>,
//! }
//!
//! struct BaseMeta<const N: usize> {
//!     buffer_lengths: [u32; N],
//!     logical_lengths: [u32; N],
//! }
//!
//! struct ExtendedMeta<const N: usize> {
//!     ranks: [u32; N],
//!     shape_offsets: [usize; N],
//!     stride_offsets: [usize; N],
//!     shapes: Vec<u32>,
//!     strides: Vec<u32>
//! }
//! ```
//! where `Vec` isn't an actual `Vec`, just a dynamically sized series of values.
//!
//! Ranks and lengths have a constant offset, while shapes/strides involve loading the tensor's
//! offset, then adding `dim` to the offset to get each shape/stride.

use cubecl_runtime::server::MetadataBinding;

// Metadata
const BUFFER_LEN: u32 = 0;
const LENGTH: u32 = 1;
const BASE_LEN: u32 = 2;

// Extended Metadata
const RANK: u32 = 0;
const SHAPE_OFFSETS: u32 = 1;
const STRIDE_OFFSETS: u32 = 2;
const EXTENDED_LEN: u32 = 3;

/// Helper to calculate metadata offsets based on buffer count and position
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    num_meta: u32,
    num_extended_meta: u32,
}

impl Metadata {
    pub fn new(num_meta: u32, num_extended_meta: u32) -> Self {
        Self {
            num_meta,
            num_extended_meta,
        }
    }

    fn offset_of(&self, id: u32) -> u32 {
        self.num_meta * id
    }

    fn base_len(&self) -> u32 {
        self.num_meta * BASE_LEN
    }

    pub fn static_len(&self) -> u32 {
        self.num_meta * BASE_LEN + self.num_extended_meta * EXTENDED_LEN
    }

    fn offset_of_extended(&self, id: u32) -> u32 {
        self.base_len() + self.num_extended_meta * id
    }

    pub fn buffer_len_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of(BUFFER_LEN) + buffer_idx
    }

    pub fn len_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of(LENGTH) + buffer_idx
    }

    pub fn rank_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of_extended(RANK) + buffer_idx
    }

    pub fn shape_offset_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of_extended(SHAPE_OFFSETS) + buffer_idx
    }

    pub fn stride_offset_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of_extended(STRIDE_OFFSETS) + buffer_idx
    }
}

/// Builder for a serialized metadata struct
///
/// Inputs/Outputs must be added in the same order they're defined in the bind group
#[derive(Default)]
pub struct MetadataBuilder {
    buffer_lens: Vec<u32>,
    lengths: Vec<u32>,
    ranks: Vec<u32>,
    shapes: Vec<Vec<u32>>,
    strides: Vec<Vec<u32>>,
}

impl MetadataBuilder {
    /// Add an array to a builder
    pub fn with_array(&mut self, buffer_len: u32, len: u32) {
        self.buffer_lens.push(buffer_len);
        self.lengths.push(len);
    }

    /// Add a tensor to a builder
    pub fn with_tensor(
        &mut self,
        rank: u32,
        buffer_len: u32,
        len: u32,
        shape: Vec<u32>,
        strides: Vec<u32>,
    ) {
        self.buffer_lens.push(buffer_len);
        self.lengths.push(len);
        self.ranks.push(rank);
        self.shapes.push(shape);
        self.strides.push(strides);
    }

    /// Build the final serialized metadata struct
    pub fn finish(self) -> MetadataBinding {
        let mut meta = self.buffer_lens;
        meta.extend(self.lengths);
        meta.extend(self.ranks.clone());

        let num_ext = self.ranks.len();
        let mut shape_offsets = Vec::with_capacity(num_ext);
        let mut stride_offsets = Vec::with_capacity(num_ext);

        let mut current_offset = meta.len() + num_ext * 2; // Total fields in static portion

        for shape in self.shapes.iter() {
            shape_offsets.push(current_offset as u32);
            current_offset += shape.len();
        }

        meta.extend(shape_offsets);

        for stride in self.strides.iter() {
            stride_offsets.push(current_offset as u32);
            current_offset += stride.len();
        }

        meta.extend(stride_offsets);

        let static_len = meta.len();

        meta.extend(self.shapes.into_iter().flatten());
        meta.extend(self.strides.into_iter().flatten());

        MetadataBinding::new(meta, static_len)
    }
}
