use cubecl::prelude::*;
use cubecl_common::quant::scheme::*;
use cubecl_common::{e2m1x2, e4m3, e5m2};
use cubecl_core as cubecl;

/// Dequantize a line of values, where `line_size * num_quants` is a power of two.
/// Unaligned values can't be dequantized in place.
#[cube]
pub fn dequantize_aligned<Q: CubePrimitive, S: CubePrimitive, F: Numeric>(
    value: Line<Q>,
    scale: S,
    #[comptime] scheme: QuantScheme,
) -> Line<F> {
    let q_values = match scheme.store {
        QuantStore::Native | QuantStore::PackedNative(_) => Line::<F>::cast_from(value),
        QuantStore::PackedU32(_) => unpack_cast_u32::<F>(Line::cast_from(value), scheme),
    };
    let scale = Line::<F>::cast_from(scale);

    match scheme.mode {
        QuantMode::Symmetric => q_values * scale,
    }
}

/// Unpack a set of values from u32, and convert to the specified floating point format.
#[cube]
pub fn unpack_cast_u32<F: Numeric>(value: Line<u32>, #[comptime] scheme: QuantScheme) -> Line<F> {
    let num_quants = scheme.num_quants();
    let native_packing = scheme.native_packing();
    let out_line_size = value.line_size().comptime() * num_quants;
    let size_bits = scheme.size_bits_value();
    let mask = comptime![packing_mask(scheme)];

    let mut out = Line::<F>::empty(out_line_size);

    #[unroll]
    for line_idx in 0..value.line_size() {
        let packed_val = value[line_idx];
        let out_offset = line_idx * num_quants;
        #[unroll]
        for packed_idx in range_stepped(0, num_quants, native_packing) {
            let shift = packed_idx * size_bits;
            let value = (packed_val >> shift as u32) & mask;

            let float_value = cast_masked::<F>(value, scheme);

            #[unroll]
            for native_idx in 0..native_packing {
                let out_offset = out_offset + packed_idx + native_idx;
                out[out_offset] = float_value[native_idx];
            }
        }
    }

    out
}

/// The mask required for each packed value, taking into account the native packing required for
/// `e2m1`.
fn packing_mask(scheme: QuantScheme) -> u32 {
    let bits = match scheme.value {
        QuantValue::E2M1 => 8, // Packed conversion
        other => other.size_bits(),
    };
    (1u32 << bits) - 1
}

/// Cast a masked-out value in the low `n` bits of a `u32` to the specified float type.
/// Applies sign conversion for integer quantization before casting to the float type,
/// while minifloats are simply truncated to `u8`, reinterpreted and then cast.
/// For `e2m1`, casting is done on the packed `e2m1x2` representation.
///
/// # Returns
/// Two floating point numbers for `e2m1`, one for all other formats.
#[cube]
fn cast_masked<F: Numeric>(value: u32, #[comptime] scheme: QuantScheme) -> Line<F> {
    match scheme.value {
        // For minifloat we can assume if they're supported then u8 is supported
        QuantValue::E5M2 => Line::<F>::cast_from(e5m2::from_bits(value as u8)),
        QuantValue::E4M3 => Line::<F>::cast_from(e4m3::from_bits(value as u8)),
        QuantValue::E2M1 => Line::<F>::cast_from(e2m1x2::from_bits(value as u8)),
        QuantValue::Q8F
        | QuantValue::Q4F
        | QuantValue::Q2F
        | QuantValue::Q8S
        | QuantValue::Q4S
        | QuantValue::Q2S => {
            let size_quant = scheme.size_bits_value() as u32;
            let sign_bit = 1u32 << (size_quant - 1);
            let two_pow_n = 1 << size_quant;

            // Branchless two's complement conversion
            // If raw >= 2^(n-1), then result = raw - 2^n
            let raw_i32 = value as i32;
            let is_negative = (value >= sign_bit) as i32; // 1 if negative, 0 if positive
            let signed_value = raw_i32 - (is_negative * two_pow_n);
            Line::<F>::cast_from(signed_value)
        }
    }
}
