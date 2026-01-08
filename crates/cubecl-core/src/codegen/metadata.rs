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

use bytemuck::cast_slice_mut;
use cubecl_ir::StorageType;
use cubecl_runtime::server::MetadataBinding;

use crate::prelude::InputScalar;

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
pub struct MetadataBuilder {
    buffer_lens: Vec<InputScalar>,
    lengths: Vec<InputScalar>,
    ranks: Vec<InputScalar>,
    shapes: Vec<Vec<InputScalar>>,
    strides: Vec<Vec<InputScalar>>,

    address_type: StorageType,
}

impl MetadataBuilder {
    pub fn new(address_type: StorageType) -> Self {
        Self {
            buffer_lens: Default::default(),
            lengths: Default::default(),
            ranks: Default::default(),
            shapes: Default::default(),
            strides: Default::default(),
            address_type,
        }
    }

    /// Add an array to a builder
    pub fn with_array(&mut self, buffer_len: u64, len: u64) {
        self.buffer_lens
            .push(InputScalar::new(buffer_len, self.address_type));
        self.lengths.push(InputScalar::new(len, self.address_type));
    }

    /// Add a tensor to a builder
    pub fn with_tensor(
        &mut self,
        rank: u64,
        buffer_len: u64,
        len: u64,
        shape: Vec<u64>,
        strides: Vec<u64>,
    ) {
        self.buffer_lens
            .push(InputScalar::new(buffer_len, self.address_type));
        self.lengths.push(InputScalar::new(len, self.address_type));
        self.ranks.push(InputScalar::new(rank, self.address_type));
        self.shapes.push(
            shape
                .into_iter()
                .map(|s| InputScalar::new(s, self.address_type))
                .collect(),
        );
        self.strides.push(
            strides
                .into_iter()
                .map(|s| InputScalar::new(s, self.address_type))
                .collect(),
        );
    }

    /// Build the final serialized metadata struct
    pub fn finish(self) -> MetadataBinding {
        let addr_size = self.address_type.size();
        let mut meta = self
            .buffer_lens
            .iter()
            .flat_map(|it| it.as_bytes())
            .collect::<Vec<_>>();

        meta.extend(self.lengths.iter().flat_map(|it| it.as_bytes()));
        meta.extend(self.ranks.iter().flat_map(|it| it.as_bytes()));

        let num_ext = self.ranks.len();
        let mut shape_offsets = Vec::with_capacity(num_ext * addr_size);
        let mut stride_offsets = Vec::with_capacity(num_ext * addr_size);

        let mut current_offset = meta.len() / addr_size + num_ext * 2; // Total fields in static portion

        for shape in self.shapes.iter() {
            let offset = InputScalar::new(current_offset, self.address_type);
            shape_offsets.extend(offset.as_bytes());
            current_offset += shape.len();
        }

        meta.extend(shape_offsets);

        for stride in self.strides.iter() {
            let offset = InputScalar::new(current_offset, self.address_type);
            stride_offsets.extend(offset.as_bytes());
            current_offset += stride.len();
        }

        meta.extend(stride_offsets);

        let static_len = meta.len() / addr_size;

        meta.extend(self.shapes.iter().flatten().flat_map(|it| it.as_bytes()));
        meta.extend(
            self.strides
                .into_iter()
                .flatten()
                .flat_map(|it| it.as_bytes()),
        );

        let total_size_64 = meta.len().div_ceil(size_of::<u64>());
        let mut meta_64 = vec![0u64; total_size_64];
        cast_slice_mut::<u64, u8>(&mut meta_64)[..meta.len()].copy_from_slice(&meta);

        MetadataBinding::new(meta_64, static_len)
    }
}
