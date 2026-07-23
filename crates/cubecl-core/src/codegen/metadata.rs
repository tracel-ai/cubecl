//! Metadata helpers to easily get offsets etc.
//!
//! Conceptually, metadata is represented like this:
//! ```ignore
//! struct Metadata<const NUM_BUFS: usize, const NUM_EXT: usize> {
//!     base: BaseMeta<NUM_BUFS>,
//!     extended: ExtendedMeta<NUM_EXT>,
//! }
//!
//! struct BaseMeta<const N: usize> {
//!     buffer_lengths: [usize; N],
//! }
//!
//! struct ExtendedMeta<const N: usize> {
//!     shape_offsets: [usize; N],
//!     stride_offsets: [usize; N],
//!     shapes: [usize],
//!     strides: [usize]
//! }
//! ```
//! where `Vec` isn't an actual `Vec`, just a dynamically sized series of values.
//!
//! Ranks and lengths have a constant offset, while shapes/strides involve loading the tensor's
//! offset, then adding `dim` to the offset to get each shape/stride.

use alloc::vec::Vec;
use bytemuck::Pod;
use cubecl_ir::{
    AddressType,
    metadata::{METADATA_BASE_LEN, METADATA_EXT_LEN},
};
use cubecl_zspace::{Shape, Strides};
use num_traits::NumCast;

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
    shapes: Vec<T>,
    strides: Vec<T>,

    offsets: Vec<usize>,
}

impl MetadataBuilder {
    /// Add an array to a builder
    pub fn register_buffer(&mut self, buffer_len: u64, address_type: AddressType) {
        match address_type {
            AddressType::U64 => {
                self.state_64.buffer_lens.push(buffer_len);
            }
            AddressType::U32 => {
                self.state_32.buffer_lens.push(buffer_len as u32);
            }
        }
    }

    /// Add a tensor to a builder
    pub fn register_tensor(
        &mut self,
        buffer_len: u64,
        shape: Shape,
        strides: Strides,
        address_type: AddressType,
    ) {
        match address_type {
            AddressType::U64 => {
                let state = &mut self.state_64;
                state.buffer_lens.push(buffer_len);
                state.offsets.push(state.shapes.len());
                state.shapes.extend(shape.iter().map(|s| *s as u64));
                state.strides.extend(strides.iter().map(|s| *s as u64));
            }
            AddressType::U32 => {
                let state = &mut self.state_32;
                state.buffer_lens.push(buffer_len as u32);
                state.offsets.push(state.shapes.len());
                state.shapes.extend(shape.iter().map(|s| *s as u32));
                state.strides.extend(strides.iter().map(|s| *s as u32));
            }
        }
    }

    pub fn static_len(&self, address_type: AddressType) -> usize {
        let (base, ext) = match address_type {
            AddressType::U32 => (self.state_32.buffer_lens.len(), self.state_32.offsets.len()),
            AddressType::U64 => (self.state_64.buffer_lens.len(), self.state_64.offsets.len()),
        };
        base * METADATA_BASE_LEN + ext * METADATA_EXT_LEN
    }

    pub fn dynamic_len(&self, address_type: AddressType) -> usize {
        match address_type {
            AddressType::U32 => self.state_32.shapes.len() + self.state_32.strides.len(),
            AddressType::U64 => self.state_64.shapes.len() + self.state_64.strides.len(),
        }
    }

    /// Build the final serialized metadata struct
    pub fn finish(&mut self, address_type: AddressType, out: (&mut [u64], &mut [u64])) {
        fn finish_inner<T: Pod + NumCast>(state: &mut State<T>, out: (&mut [u64], &mut [u64])) {
            let mut sized = bytemuck::cast_slice_mut::<u64, u8>(out.0);
            let mut dynamic = bytemuck::cast_slice_mut::<u64, u8>(out.1);

            {
                let buffer_lens = bytemuck::cast_slice::<T, u8>(&state.buffer_lens);

                sized[..buffer_lens.len()].copy_from_slice(buffer_lens);
                sized = &mut sized[buffer_lens.len()..];
            }

            state.buffer_lens.clear();

            let strides_offset_base = state.shapes.len();

            for offs in state.offsets.iter() {
                let offset = [T::from(*offs).unwrap()];
                let bytes = bytemuck::cast_slice(&offset);
                sized[..bytes.len()].copy_from_slice(bytes);
                sized = &mut sized[size_of::<T>()..];
            }

            for offs in state.offsets.drain(..) {
                let offset = [T::from(strides_offset_base + offs).unwrap()];
                let bytes = bytemuck::cast_slice(&offset);
                sized[..bytes.len()].copy_from_slice(bytes);
                sized = &mut sized[size_of::<T>()..];
            }

            {
                let shapes = bytemuck::cast_slice::<T, u8>(&state.shapes);
                let strides = bytemuck::cast_slice::<T, u8>(&state.strides);

                dynamic[..shapes.len()].copy_from_slice(shapes);
                dynamic = &mut dynamic[shapes.len()..];

                dynamic[..strides.len()].copy_from_slice(strides);
            }

            state.shapes.clear();
            state.strides.clear();
        }

        match address_type {
            AddressType::U32 => finish_inner(&mut self.state_32, out),
            AddressType::U64 => finish_inner(&mut self.state_64, out),
        }
    }
}
