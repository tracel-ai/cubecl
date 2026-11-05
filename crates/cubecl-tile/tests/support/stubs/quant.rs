use cubecl::quant::scheme::QuantMode;
use cubecl_common::{
    e2m1, e4m3, e5m2,
    quant::scheme::{QuantScheme, QuantStore, QuantValue},
};

pub fn quantize(
    values: &[f32],
    shape: &[usize],
    scales: &[f32],
    block_dims: &[usize],
    scheme: &QuantScheme,
) -> Vec<u8> {
    let scales_shape = scales_grid(shape, block_dims);
    let quantized = quantized_values(values, shape, scales, block_dims, &scales_shape, scheme);

    match scheme.store {
        QuantStore::Native => encode_native(&quantized, scheme.value),
        QuantStore::PackedU32(_) => encode_packed_u32(&quantized, scheme),
        other => panic!("quantize stub: unsupported store {other:?}"),
    }
}

pub fn dequantize(
    bytes: &[u8],
    shape: &[usize],
    scales: &[f32],
    block_dims: &[usize],
    scheme: &QuantScheme,
) -> Vec<f32> {
    let scales_shape = scales_grid(shape, block_dims);

    let quants = match scheme.store {
        QuantStore::Native => decode_native(bytes, scheme.value),
        QuantStore::PackedU32(_) => decode_packed_u32(bytes, scheme),
        other => panic!("dequantize stub: unsupported store {other:?}"),
    };

    dequantized_values(&quants, scales, shape, block_dims, &scales_shape, scheme)
}

fn quantized_values(
    values: &[f32],
    shape: &[usize],
    scales: &[f32],
    block_dims: &[usize],
    scales_shape: &[usize],
    scheme: &QuantScheme,
) -> Vec<f32> {
    let (range_min, range_max) = scheme.value.range();

    match scheme.mode {
        QuantMode::Symmetric => values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let scale = scales[scale_index(i, shape, block_dims, scales_shape)];
                let scaled = v / scale;
                let snapped = if rounds_to_integers(scheme.value) {
                    scaled.round()
                } else {
                    scaled
                };
                snapped.clamp(range_min, range_max)
            })
            .collect(),
        // A lookup field is a table index, not a rounded value; these stubs carry no table, so
        // a lookup test states its indices exactly ([`pack_q_values`]) and multiplies the table into
        // its own host reference.
        QuantMode::Lookup => panic!("quantize stub: lookup takes exact indices via pack_q_values"),
    }
}

/// Whether the format's representable values are the integers in its range, which is what makes
/// rounding to a whole number here the right rounding for it.
///
/// A minifloat's grid is not the integers — `e2m1`
/// holds `0.5` and `1.5` — so rounding one here would quantize it twice, and this reference would
/// then demand of the kernel exactly the degradation the format exists to avoid. Those formats
/// round where they are encoded, against the codes they actually have.
fn rounds_to_integers(quant: QuantValue) -> bool {
    match quant {
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => true,
        QuantValue::E5M2 | QuantValue::E4M3 | QuantValue::E2M1 => false,
    }
}

fn dequantized_values(
    quantized: &[f32],
    scales: &[f32],
    shape: &[usize],
    block_dims: &[usize],
    scales_shape: &[usize],
    scheme: &QuantScheme,
) -> Vec<f32> {
    match scheme.mode {
        QuantMode::Symmetric => quantized
            .iter()
            .enumerate()
            .map(|(i, &q)| q * scales[scale_index(i, shape, block_dims, scales_shape)])
            .collect(),
        QuantMode::Lookup => panic!(
            "dequantize stub: lookup carries no table here; fold table[index] * scale into the \
             test's own host reference"
        ),
    }
}

/// 1:1 native storage: `i8` for `Q8*`, fp8 bits for `E4M3`/`E5M2`.
fn encode_native(quantized: &[f32], value: QuantValue) -> Vec<u8> {
    match value {
        QuantValue::Q8F | QuantValue::Q8S => quantized.iter().map(|&q| q as i8 as u8).collect(),
        QuantValue::E4M3 => quantized
            .iter()
            .map(|&q| e4m3::from_f32(q).to_bits())
            .collect(),
        QuantValue::E5M2 => quantized
            .iter()
            .map(|&q| e5m2::from_f32(q).to_bits())
            .collect(),
        other => panic!("quantize stub: {other:?} is not supported for native storage"),
    }
}

fn decode_native(bytes: &[u8], value: QuantValue) -> Vec<f32> {
    match value {
        QuantValue::Q8F | QuantValue::Q8S => bytes.iter().map(|&b| b as i8 as f32).collect(),
        QuantValue::E4M3 => bytes.iter().map(|&b| e4m3::from_bits(b).to_f32()).collect(),
        QuantValue::E5M2 => bytes.iter().map(|&b| e5m2::from_bits(b).to_f32()).collect(),
        other => panic!("dequantize stub: {other:?} is not supported for native storage"),
    }
}

/// [`encode_packed_u32`] over exact fields, as `u32` words: for tests that need bit-exact control
/// of the packed payload rather than the quantize stub's rounding.
///
/// The fields are taken as given, so under a minifloat scheme they are that format's *codes*, not
/// the magnitudes they stand for.
pub fn pack_q_values(q: &[i32], scheme: &QuantScheme) -> Vec<u32> {
    let size_quant = scheme.size_bits_value();
    let num_quants = scheme.num_quants();
    let mask = quant_mask(size_quant);
    assert!(
        q.len().is_multiple_of(num_quants),
        "pack_q_values: {} values do not fill whole {num_quants}-value words",
        q.len()
    );
    q.chunks(num_quants)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0u32, |acc, (p, &v)| {
                acc | ((v as u32 & mask) << (p * size_quant))
            })
        })
        .collect()
}

/// Pack `num_quants` consecutive quants (along the innermost dimension) into
/// each `u32`, low bits first, matching `pack_q`.
fn encode_packed_u32(quantized: &[f32], scheme: &QuantScheme) -> Vec<u8> {
    let size_quant = scheme.size_bits_value();
    let num_quants = scheme.num_quants();
    let mask = quant_mask(size_quant);

    let packed: Vec<u32> = quantized
        .chunks(num_quants)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0u32, |acc, (p, &q)| {
                let bits = field_code(q, scheme.value) & mask;
                acc | (bits << (p * size_quant))
            })
        })
        .collect();

    bytemuck::cast_slice(&packed).to_vec()
}

/// The field `q` occupies: the format's *code* for it, which is the value itself only for the
/// integer formats.
///
/// A minifloat encoded by casting would leave the reference reconstructing on the integer grid —
/// `e2m1` losing `0.5` and `1.5`, `e4m3` wrapping its codes above ±127 — and a kernel that made
/// the same mistake would then match it.
fn field_code(q: f32, value: QuantValue) -> u32 {
    match value {
        QuantValue::E2M1 => e2m1::from_f32(q).to_bits() as u32,
        QuantValue::E4M3 => e4m3::from_f32(q).to_bits() as u32,
        QuantValue::E5M2 => e5m2::from_f32(q).to_bits() as u32,
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => q as i32 as u32,
    }
}

/// Inverse of [`encode_packed_u32`]: unpack each `u32` into `num_quants` quant values, matching
/// `unpack_q`.
fn decode_packed_u32(bytes: &[u8], scheme: &QuantScheme) -> Vec<f32> {
    let size_quant = scheme.size_bits_value();
    let num_quants = scheme.num_quants();
    let mask = quant_mask(size_quant);

    let mut out = Vec::with_capacity((bytes.len() / 4) * num_quants);
    for &word in bytemuck::cast_slice::<u8, u32>(bytes) {
        for p in 0..num_quants {
            let raw = (word >> (p * size_quant)) & mask;
            out.push(code_value(raw, size_quant, scheme.value));
        }
    }
    out
}

/// The value a field stands for: the inverse of [`field_code`]. A minifloat code is decoded by its
/// format, and only an integer format's field is a two's complement number.
fn code_value(raw: u32, size_quant: usize, value: QuantValue) -> f32 {
    match value {
        QuantValue::E2M1 => e2m1::from_bits(raw as u8).to_f32(),
        QuantValue::E4M3 => e4m3::from_bits(raw as u8).to_f32(),
        QuantValue::E5M2 => e5m2::from_bits(raw as u8).to_f32(),
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => {
            let mask = quant_mask(size_quant);
            let sign_bit = 1u32 << (size_quant - 1);
            // Two's-complement sign extension.
            let q = if raw & sign_bit != 0 {
                (raw | !mask) as i32
            } else {
                raw as i32
            };
            q as f32
        }
    }
}

fn quant_mask(size_quant: usize) -> u32 {
    if size_quant >= 32 {
        u32::MAX
    } else {
        (1u32 << size_quant) - 1
    }
}

/// Refuse a scheme this reference cannot stand in for.
///
/// It quantizes with one scale per value, so a two-level scheme would give a reference that
/// ignores the outer factor. A kernel that drops it too would then agree with the reference, and
/// the test would pass with both of them wrong.
pub(crate) fn assert_supported(scheme: &QuantScheme) {
    assert!(
        scheme.num_levels() <= 1,
        "two-level quantization is not supported by the reference quantizer, got {scheme:?}"
    );
}

/// The scheme's per-axis block edges over `shape`: the number of elements each scale covers
/// along each axis. A scheme with no block level has one scale for the whole tensor, so its
/// block spans everything.
pub(crate) fn block_dims(scheme: &QuantScheme, shape: &[usize]) -> Vec<usize> {
    assert_supported(scheme);

    match scheme.block_size() {
        Some(block) => block
            .to_dim_vec(shape.len())
            .into_iter()
            .map(usize::from)
            .collect(),
        None => shape.to_vec(),
    }
}

/// The block scale grid: each axis divided by its block extent, so one entry per block.
pub fn scales_grid(shape: &[usize], block_dims: &[usize]) -> Vec<usize> {
    assert_eq!(
        shape.len(),
        block_dims.len(),
        "shape/block_dims rank mismatch"
    );
    shape
        .iter()
        .zip(block_dims)
        .map(|(&dim, &block)| {
            assert!(
                block > 0 && dim.is_multiple_of(block),
                "axis of {dim} elements does not divide into {block}-element blocks"
            );
            dim / block
        })
        .collect()
}

/// Map a logical (row-major) element index to the index of its block in the
/// row-major scales grid.
fn scale_index(
    linear: usize,
    shape: &[usize],
    block_dims: &[usize],
    scales_shape: &[usize],
) -> usize {
    let rank = shape.len();
    let mut rem = linear;
    let mut coord = vec![0usize; rank];
    for d in (0..rank).rev() {
        coord[d] = rem % shape[d];
        rem /= shape[d];
    }

    let mut block_linear = 0;
    for d in 0..rank {
        block_linear = block_linear * scales_shape[d] + coord[d] / block_dims[d];
    }
    block_linear
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::quant::scheme::{QuantMode, QuantStore, QuantValue, ScaleDtype};

    fn scheme(value: QuantValue, store: QuantStore) -> QuantScheme {
        QuantScheme::default()
            .with_mode(QuantMode::Symmetric)
            .with_value(value)
            .with_store(store)
    }

    /// Quantize then dequantize and assert the round-trip stays within one
    /// quantization step (`scale`), the worst-case symmetric error.
    fn assert_round_trips(value: QuantValue, store: QuantStore, scale: f32) {
        let s = scheme(value, store).per_tensor(ScaleDtype::F32);
        let n = 64; // multiple of every num_quants we test (≤16)
        let values: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect();

        let bytes = quantize(&values, &[n], &[scale], &[n], &s);
        let restored = dequantize(&bytes, &[n], &[scale], &[n], &s);

        let max_err = scale + 1e-6;
        for (got, want) in restored.iter().zip(&values) {
            assert!(
                (got - want).abs() <= max_err,
                "{value:?}/{store:?}: dequant {got} too far from {want}",
            );
        }
    }

    #[test]
    fn native_q8_is_one_to_one_i8() {
        let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let scale = 1.0 / 127.0;
        let bytes = quantize(
            &values,
            &[values.len()],
            &[scale],
            &[values.len()],
            &scheme(QuantValue::Q8S, QuantStore::Native).per_tensor(ScaleDtype::F32),
        );

        let got: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(got, vec![-127, -64, 0, 64, 127]);
    }

    #[test]
    fn round_trips_for_integer_schemes() {
        assert_round_trips(QuantValue::Q8S, QuantStore::Native, 1.0 / 127.0);
        assert_round_trips(QuantValue::Q8S, QuantStore::PackedU32(0), 1.0 / 127.0);
        assert_round_trips(QuantValue::Q4S, QuantStore::PackedU32(0), 1.0 / 7.0);
        assert_round_trips(QuantValue::Q2S, QuantStore::PackedU32(0), 1.0 / 1.0);
    }

    #[test]
    fn round_trips_for_fp8_native() {
        // Like the kernel, fp8 quantization scales and then encodes, with no rounding of its own:
        // the only error the round trip carries is the format's own representation error.
        assert_round_trips(QuantValue::E4M3, QuantStore::Native, 1.0 / 8.0);
        assert_round_trips(QuantValue::E5M2, QuantStore::Native, 1.0 / 4.0);
    }

    /// Quantize then dequantize `values` at a scale of one, so what comes back is the codec's
    /// answer and nothing else.
    fn round_trip_at_unit_scale(value: QuantValue, store: QuantStore, values: &[f32]) -> Vec<f32> {
        let s = scheme(value, store).per_tensor(ScaleDtype::F32);
        let n = values.len();
        let bytes = quantize(values, &[n], &[1.0], &[n], &s);

        dequantize(&bytes, &[n], &[1.0], &[n], &s)
    }

    /// A packed minifloat field is the format's code, so every value the format holds comes back as
    /// itself — the fractions included. The integer codec this replaced returned `1.0` for `0.5`
    /// and `2.0` for `1.5`, having rounded both before they ever reached the format.
    #[test]
    fn packed_e2m1_holds_the_format_grid() {
        let values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -6.0];
        let restored =
            round_trip_at_unit_scale(QuantValue::E2M1, QuantStore::PackedU32(0), &values);

        assert_eq!(
            restored, values,
            "an e2m1 code point did not come back as itself"
        );
    }

    /// The same for `e4m3`, whose codes reach ±448 while a byte read as two's complement stops at
    /// ±127: a field cast from the value rather than encoded wraps rather than merely losing
    /// precision.
    #[test]
    fn packed_e4m3_holds_fractions_and_large_magnitudes() {
        let values = [0.5f32, 1.5, -0.5, 0.0625, 288.0, -288.0, 256.0, 144.0];
        let restored =
            round_trip_at_unit_scale(QuantValue::E4M3, QuantStore::PackedU32(0), &values);

        assert_eq!(
            restored, values,
            "an e4m3 code point did not come back as itself"
        );
    }

    #[test]
    fn block_level_uses_per_block_scale() {
        // Two blocks of 4 along the last dim, each with its own scale.
        let s = scheme(QuantValue::Q8S, QuantStore::Native).per_block([4u8], ScaleDtype::F32);
        let values = vec![
            0.1, 0.2, 0.3, 0.4, // block 0
            10.0, 20.0, 30.0, 40.0, // block 1
        ];
        let scales = vec![0.4 / 127.0, 40.0 / 127.0];

        let bytes = quantize(&values, &[8], &scales, &[4], &s);
        let got: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();

        // Largest element of each block saturates to the quant max.
        assert_eq!(got[3], 127);
        assert_eq!(got[7], 127);

        // And the per-block scale is applied on the way back.
        let restored = dequantize(&bytes, &[8], &scales, &[4], &s);
        for (got, want) in restored.iter().zip(&values) {
            let step = if *want < 1.0 { scales[0] } else { scales[1] };
            assert!((got - want).abs() <= step + 1e-6);
        }
    }
}
