//! Software `e2m1` conversion, kept beside the other minifloat codecs in
//! [`cubecl_core::post_processing::fp4`] and re-exported here for the quantized readers.

pub use cubecl_core::post_processing::fp4::{
    e2m1_bits_to_float, e2m1_packed_bits_to_float, float_to_e2m1_bits,
};
