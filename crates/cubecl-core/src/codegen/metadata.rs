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

use alloc::{vec, vec::Vec};
use bytemuck::Pod;
use cubecl_ir::AddressType;
use cubecl_runtime::server::MetadataBinding;
use num_traits::NumCast;

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
    state_32: State<u32>,
    state_64: State<u64>,
}

#[derive(Default)]
struct State<T: Pod> {
    buffer_lens: Vec<T>,
    lengths: Vec<T>,
    ranks: Vec<T>,
    shapes: Vec<T>,
    strides: Vec<T>,

    offsets: Vec<usize>,
}

impl MetadataBuilder {
    /// Add an array to a builder
    pub fn register_array(&mut self, buffer_len: u64, len: u64, address_type: AddressType) {
        match address_type {
            AddressType::U64 => {
                self.state_64.buffer_lens.push(buffer_len);
                self.state_64.lengths.push(len);
            }
            AddressType::U32 => {
                self.state_32.buffer_lens.push(buffer_len as u32);
                self.state_32.lengths.push(len as u32);
            }
        }
    }

    /// Add a tensor to a builder
    pub fn register_tensor(
        &mut self,
        rank: u64,
        buffer_len: u64,
        len: u64,
        shape: &[usize],
        strides: &[usize],
        address_type: AddressType,
    ) {
        match address_type {
            AddressType::U64 => {
                let state = &mut self.state_64;
                state.buffer_lens.push(buffer_len);
                state.lengths.push(len);
                state.ranks.push(rank);
                state.offsets.push(state.shapes.len());
                state.shapes.extend(shape.iter().map(|s| *s as u64));
                state.strides.extend(strides.iter().map(|s| *s as u64));
            }
            AddressType::U32 => {
                let state = &mut self.state_32;
                state.buffer_lens.push(buffer_len as u32);
                state.lengths.push(len as u32);
                state.ranks.push(rank as u32);
                state.offsets.push(state.shapes.len());
                state.shapes.extend(shape.iter().map(|s| *s as u32));
                state.strides.extend(strides.iter().map(|s| *s as u32));
            }
        }
    }

    /// Build the final serialized metadata struct
    pub fn finish(&mut self, address_type: AddressType) -> MetadataBinding {
        fn finish_inner<T: Pod + NumCast>(state: &mut State<T>) -> MetadataBinding {
            let num_base = state.buffer_lens.len();
            let num_ext = state.ranks.len();

            // All entries have buffer_len and len, extended also have rank, shape_offs, strides_offs
            let static_len = num_base * BASE_LEN as usize + num_ext * EXTENDED_LEN as usize;
            let dynamic_len = state.shapes.len() + state.strides.len();
            let total_len = static_len + dynamic_len;

            let len_u64 = (total_len * size_of::<T>()).div_ceil(size_of::<u64>());
            let mut meta_64 = vec![0u64; len_u64];
            let mut meta = bytemuck::cast_slice_mut::<u64, u8>(&mut meta_64);

            {
                let buffer_lens = bytemuck::cast_slice::<T, u8>(&state.buffer_lens);
                let lengths = bytemuck::cast_slice::<T, u8>(&state.lengths);
                let ranks = bytemuck::cast_slice::<T, u8>(&state.ranks);

                meta[..buffer_lens.len()].copy_from_slice(buffer_lens);
                meta = &mut meta[buffer_lens.len()..];

                meta[..lengths.len()].copy_from_slice(lengths);
                meta = &mut meta[lengths.len()..];

                meta[..ranks.len()].copy_from_slice(ranks);
                meta = &mut meta[ranks.len()..];
            }

            state.buffer_lens.clear();
            state.lengths.clear();
            state.ranks.clear();

            let shape_offset_base = static_len;
            let strides_offset_base = shape_offset_base + state.shapes.len();

            for offs in state.offsets.iter() {
                let offset = [T::from(shape_offset_base + *offs).unwrap()];
                let bytes = bytemuck::cast_slice(&offset);
                meta[..bytes.len()].copy_from_slice(bytes);
                meta = &mut meta[size_of::<T>()..];
            }

            for offs in state.offsets.drain(..) {
                let offset = [T::from(strides_offset_base + offs).unwrap()];
                let bytes = bytemuck::cast_slice(&offset);
                meta[..bytes.len()].copy_from_slice(bytes);
                meta = &mut meta[size_of::<T>()..];
            }

            {
                let shapes = bytemuck::cast_slice::<T, u8>(&state.shapes);
                let strides = bytemuck::cast_slice::<T, u8>(&state.strides);

                meta[..shapes.len()].copy_from_slice(shapes);
                meta = &mut meta[shapes.len()..];

                meta[..strides.len()].copy_from_slice(strides);
            }

            state.shapes.clear();
            state.strides.clear();

            MetadataBinding::new(meta_64, static_len)
        }

        match address_type {
            AddressType::U32 => finish_inner(&mut self.state_32),
            AddressType::U64 => finish_inner(&mut self.state_64),
        }
    }
}
