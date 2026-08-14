use cubecl::prelude::*;
use cubecl_common::quant::scheme::*;
use cubecl_common::{e2m1x2, e4m3, e5m2};
use cubecl_core as cubecl;

/// Dequantize a vector of values, where `vector_size * num_quants` is a power of two.
/// Unaligned values can't be dequantized in place.
///
/// `scale` is the effective scale for these values: how many scale levels the scheme has and how
/// they combine is the caller's business, folded before the call. This is what keeps the
/// primitive per-read arithmetic, serving equally a one-level read, a view folding an outer level
/// per position, or a tile handing every read one register.
#[cube]
pub fn dequantize_aligned<Q: Scalar, S: CubePrimitive, F: Numeric, NQ: Size, NF: Size>(
    value: Vector<Q, NQ>,
    scale: S,
    #[comptime] scheme: QuantScheme,
) -> Vector<F, NF> {
    let q_values = match scheme.store {
        QuantStore::Native | QuantStore::PackedNative(_) => Vector::<F, NF>::cast_from(value),
        QuantStore::PackedU32(_) => unpack_cast_u32::<F, NQ, NF>(Vector::cast_from(value), scheme),
    };

    match scheme.mode {
        QuantMode::Symmetric => q_values * Vector::<F, NF>::cast_from(scale),
    }
}

/// The effective scale of values whose outer levels multiply on top of their own.
///
/// The levels multiply in f32: an inner scale is normalized against the outer ones, so on its own
/// it overflows a narrow compute type by orders of magnitude before the outer scales can bring
/// the product back into range.
#[cube]
pub fn multiply_outer_scale<S: CubePrimitive>(outer_scale: f32, scale: S) -> f32 {
    outer_scale * f32::cast_from(scale)
}

/// Unpack a set of values from u32, and convert to the specified floating point format.
#[cube]
pub fn unpack_cast_u32<F: Numeric, NQ: Size, NF: Size>(
    value: Vector<u32, NQ>,
    #[comptime] scheme: QuantScheme,
) -> Vector<F, NF> {
    let num_quants = scheme.num_quants();
    let native_packing = scheme.native_packing();
    let size_bits = scheme.size_bits_value();
    let mask = comptime![packing_mask(scheme)];
    let size!(NP) = native_packing;

    let mut out = Vector::<F, NF>::empty();

    #[unroll]
    for vector_idx in 0..value.vector_size() {
        let packed_val = value.extract(vector_idx);
        let out_offset = vector_idx * num_quants;
        #[unroll]
        for packed_idx in range_stepped(0, num_quants, native_packing) {
            let shift = packed_idx * size_bits;
            let value = (packed_val >> shift as u32) & mask;

            let float_value = cast_masked::<F, NP>(value, scheme);

            #[unroll]
            for native_idx in 0..native_packing {
                let out_offset = out_offset + packed_idx + native_idx;
                out.insert(out_offset, float_value.extract(native_idx));
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
fn cast_masked<F: Numeric, N: Size>(value: u32, #[comptime] scheme: QuantScheme) -> Vector<F, N> {
    match scheme.value {
        // For minifloat we can assume if they're supported then u8 is supported
        QuantValue::E5M2 => Vector::<F, N>::cast_from(e5m2::from_bits(value as u8)),
        QuantValue::E4M3 => Vector::<F, N>::cast_from(e4m3::from_bits(value as u8)),
        QuantValue::E2M1 => Vector::<F, N>::cast_from(e2m1x2::from_bits(value as u8)),
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
            Vector::<F, N>::cast_from(signed_value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::ir::{ElemType, Scope, UIntKind};
    use cubecl_core::{define_size, ir::settings::Dim3};

    define_size!(N1);

    /// A root scope carries no typemap; the launcher normally picks the index width.
    fn test_scope() -> Scope {
        let scope = Scope::root(KernelSettings::new(
            Dim3::new_single(),
            ExecutionMode::Checked,
            AddressType::U32,
        ));
        scope.register_size::<N1>(1);
        scope.register_type::<usize>(ElemType::UInt(UIntKind::U32));
        scope
    }

    /// The primitive is level-agnostic: it takes the effective scale for the values it unpacks,
    /// and how many levels folded into that scale is the caller's business.
    #[test]
    fn expanding_takes_one_scale_whatever_the_levels() {
        let scope = test_scope();
        let one = f32::__expand_new(&scope, 1.0);
        let value = Vector::<f32, N1>::__expand_new(&scope, one);

        for scheme in [
            QuantScheme::default(),
            QuantScheme::default().per_block([32], ScaleDtype::F32),
            QuantScheme::default()
                .per_block([32], ScaleDtype::F32)
                .per_tensor(ScaleDtype::F32),
        ] {
            dequantize_aligned::expand::<f32, f32, f32, N1, N1>(&scope, value, one, scheme);
        }
    }
}
