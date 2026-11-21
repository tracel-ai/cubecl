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

/// Which value of an argument's metadata a [`MetadataOverflow`] is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataQuantity {
    /// The number of elements of the argument's buffer.
    BufferLength,
    /// The extent of the dimension at this index.
    Shape(usize),
    /// The stride of the dimension at this index.
    Stride(usize),
}

impl core::fmt::Display for MetadataQuantity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferLength => f.write_str("buffer length"),
            Self::Shape(dim) => write!(f, "shape {dim}"),
            Self::Stride(dim) => write!(f, "stride {dim}"),
        }
    }
}

/// A metadata value that the address type it is registered at cannot hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetadataOverflow {
    /// Which value overflowed.
    pub quantity: MetadataQuantity,
    /// The value that does not fit.
    pub value: u128,
}

/// The first value registered for one argument that its address type could
/// not hold.
#[derive(Default)]
struct Overflows {
    first: Option<MetadataOverflow>,
}

impl Overflows {
    /// `value` as `T`, or zero standing in for it when `T` cannot hold it, in
    /// which case the overflow is recorded unless one came before it.
    fn fit<T: TryFrom<u128> + Default>(&mut self, value: u128, quantity: MetadataQuantity) -> T {
        T::try_from(value).unwrap_or_else(|_| {
            self.first
                .get_or_insert(MetadataOverflow { quantity, value });
            T::default()
        })
    }

    fn into_result(self) -> Result<(), MetadataOverflow> {
        match self.first {
            Some(overflow) => Err(overflow),
            None => Ok(()),
        }
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
    shapes: Vec<T>,
    strides: Vec<T>,

    offsets: Vec<usize>,
}

impl<T: Pod + TryFrom<u128> + Default> State<T> {
    fn push_buffer(&mut self, buffer_len: u64) -> Result<(), MetadataOverflow> {
        let mut overflow = Overflows::default();
        let len = overflow.fit(buffer_len.into(), MetadataQuantity::BufferLength);
        self.buffer_lens.push(len);
        overflow.into_result()
    }

    fn push_tensor(
        &mut self,
        buffer_len: u64,
        shape: &Shape,
        strides: &Strides,
    ) -> Result<(), MetadataOverflow> {
        let mut overflow = Overflows::default();
        let len = overflow.fit(buffer_len.into(), MetadataQuantity::BufferLength);
        self.buffer_lens.push(len);
        self.offsets.push(self.shapes.len());
        for (dim, extent) in shape.iter().enumerate() {
            let extent = overflow.fit(*extent as u128, MetadataQuantity::Shape(dim));
            self.shapes.push(extent);
        }
        for (dim, stride) in strides.iter().enumerate() {
            let stride = overflow.fit(*stride as u128, MetadataQuantity::Stride(dim));
            self.strides.push(stride);
        }
        overflow.into_result()
    }
}

impl MetadataBuilder {
    /// Add an array to a builder
    ///
    /// # Errors
    ///
    /// When `buffer_len` does not fit `address_type`. The builder still holds
    /// a placeholder in its place, so the metadata keeps the layout the kernel
    /// expects, but a launch with it must not run.
    pub fn register_buffer(
        &mut self,
        buffer_len: u64,
        address_type: AddressType,
    ) -> Result<(), MetadataOverflow> {
        match address_type {
            AddressType::U64 => self.state_64.push_buffer(buffer_len),
            AddressType::U32 => self.state_32.push_buffer(buffer_len),
        }
    }

    /// Add a tensor to a builder
    ///
    /// # Errors
    ///
    /// The first of `buffer_len`, `shape` and `strides` that does not fit
    /// `address_type`. The builder still holds a placeholder for every value,
    /// so the metadata keeps the layout the kernel expects, but a launch with
    /// it must not run.
    pub fn register_tensor(
        &mut self,
        buffer_len: u64,
        shape: Shape,
        strides: Strides,
        address_type: AddressType,
    ) -> Result<(), MetadataOverflow> {
        match address_type {
            AddressType::U64 => self.state_64.push_tensor(buffer_len, &shape, &strides),
            AddressType::U32 => self.state_32.push_tensor(buffer_len, &shape, &strides),
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

#[cfg(test)]
mod tests {
    use super::*;

    const PAST_U32: u64 = 1 << 32;

    /// A buffer length past `u32` is refused at a 32-bit address type and kept whole at a 64-bit
    /// one. The refused one still takes its slot, so the metadata keeps the layout the kernel
    /// expects.
    #[test]
    fn a_buffer_length_past_u32_is_refused_at_32_bits_only() {
        let mut builder = MetadataBuilder::default();
        assert_eq!(
            builder.register_buffer(PAST_U32, AddressType::U32),
            Err(MetadataOverflow {
                quantity: MetadataQuantity::BufferLength,
                value: PAST_U32.into(),
            })
        );
        assert_eq!(builder.static_len(AddressType::U32), METADATA_BASE_LEN);
        assert_eq!(builder.register_buffer(PAST_U32, AddressType::U64), Ok(()));
    }

    /// The first value of a tensor that does not fit is the one reported, and every value keeps
    /// its slot.
    #[test]
    fn a_tensor_reports_its_first_value_past_u32() {
        let mut builder = MetadataBuilder::default();
        let past = PAST_U32 as usize;
        assert_eq!(
            builder.register_tensor(2, [1, past].into(), [past, 0].into(), AddressType::U32),
            Err(MetadataOverflow {
                quantity: MetadataQuantity::Shape(1),
                value: past as u128,
            })
        );
        assert_eq!(builder.dynamic_len(AddressType::U32), 4);
        assert_eq!(
            builder.register_tensor(2, [1, past].into(), [past, 0].into(), AddressType::U64),
            Ok(())
        );
    }
}
