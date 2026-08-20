use alloc::{vec, vec::Vec};
use std::println;

use crate::{self as cubecl, as_type};
use cubecl::prelude::*;
use cubecl_common::{e2m1x2, e2m3, e3m2, e4m3, e5m2, ue8m0};
use cubecl_ir::features::TypeUsage;
use enumset::EnumSet;

#[cube(launch_unchecked)]
pub fn kernel_fp8<F: Float, N: Size>(input: &mut [Vector<F, N>], out: &mut [Vector<u8, N>]) {
    if ABSOLUTE_POS == 0 {
        let value = input[0];

        out[0] = Vector::reinterpret(Vector::<e4m3, N>::cast_from(value));
        out[1] = Vector::reinterpret(Vector::<e5m2, N>::cast_from(value));
        input[0] = Vector::cast_from(Vector::<e4m3, N>::reinterpret(out[0]));
    }
}

#[cube(launch_unchecked)]
pub fn kernel_fp6<F: Float, N: Size>(input: &mut [Vector<F, N>], out: &mut [Vector<u8, N>]) {
    if ABSOLUTE_POS == 0 {
        let value = input[0];

        out[0] = Vector::reinterpret(Vector::<e2m3, N>::cast_from(value));
        out[1] = Vector::reinterpret(Vector::<e3m2, N>::cast_from(value));
        input[0] = Vector::cast_from(Vector::<e2m3, N>::reinterpret(out[0]));
    }
}

#[cube(launch_unchecked)]
pub fn kernel_fp4<F: Float, N: Size, N2: Size>(
    input: &mut [Vector<F, N>],
    out: &mut [Vector<u8, N2>],
) {
    if ABSOLUTE_POS == 0 {
        let value = input[0];

        out[0] = Vector::reinterpret(Vector::<e2m1x2, N2>::cast_from(value));
        input[0] = Vector::cast_from(Vector::<e2m1x2, N2>::reinterpret(out[0]));
    }
}

#[cube(launch_unchecked)]
pub fn kernel_scale<N: Size>(input: &mut [Vector<f32, N>], out: &mut [Vector<ue8m0, N>]) {
    if ABSOLUTE_POS == 0 {
        let value = input[0];

        out[0] = Vector::<ue8m0, N>::cast_from(value);
        input[0] = Vector::cast_from(out[0]);
    }
}

#[allow(clippy::unusual_byte_groupings, reason = "Split by float components")]
pub fn test_fp8<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    vector_size: VectorSize,
) {
    if !e4m3::supported_uses(&client).contains(TypeUsage::Conversion) {
        println!("Unsupported, skipping");
        return;
    }

    let data = as_type![F: -2.1, 1.8, 0.4, 1.2];
    let num_out = vector_size;
    let handle1 = client.create_from_slice(F::as_bytes(&data[..num_out]));
    let handle2 = client.empty(2 * num_out * size_of::<u8>());

    unsafe {
        kernel_fp8::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            vector_size,
            BufferArg::from_raw_parts(handle1.clone(), num_out),
            BufferArg::from_raw_parts(handle2.clone(), 2 * num_out),
        )
    };

    let actual = client.read_one_unchecked(handle2);
    let actual = u8::from_bytes(&actual);
    let expect_0: Vec<u8> = vec![0b1_1000_000, 0b0_0111_110, 0b0_0101_101, 0b0_0111_010];
    let expect_1: Vec<u8> = vec![0b1_10000_00, 0b0_01111_11, 0b0_01101_10, 0b0_01111_01];
    let mut expected = expect_0[..num_out].to_vec();
    expected.extend(expect_1[..num_out].iter().copied());

    // TODO: Eventually add approx comparison that can deal with arbitrary floats. Manually
    // double check for now
    let actual_2 = client.read_one_unchecked(handle1);
    let actual_2 = F::from_bytes(&actual_2);
    println!("actual_2: {actual_2:?}");

    // Data rounded to the nearest e4m3 value
    let expected_data = as_type![F: -2.0, 1.75, 0.40625, 1.25];

    assert_eq!(actual, &expected);
    assert_eq!(&actual_2[..num_out], &expected_data[..num_out]);
}

#[allow(clippy::unusual_byte_groupings, reason = "Split by float components")]
pub fn test_fp6<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    vector_size: VectorSize,
) {
    if !e2m3::supported_uses(&client).contains(TypeUsage::Conversion) {
        println!("Unsupported, skipping");
        return;
    }

    let data = as_type![F: -2.1, 1.8, 0.4, 1.2];
    let num_out = vector_size;
    let handle1 = client.create_from_slice(F::as_bytes(&data[..num_out]));
    let handle2 = client.empty(2 * num_out * size_of::<u8>());

    unsafe {
        kernel_fp6::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            vector_size,
            BufferArg::from_raw_parts(handle1.clone(), num_out),
            BufferArg::from_raw_parts(handle2.clone(), 2 * num_out),
        )
    };

    let actual = client.read_one_unchecked(handle2);
    let actual = u8::from_bytes(&actual);
    let expect_0: Vec<u8> = vec![0b1_10_000, 0b0_01_110, 0b0_00_011, 0b0_01_010];
    let expect_1: Vec<u8> = vec![0b1_100_00, 0b0_011_11, 0b0_001_10, 0b0_011_01];
    let mut expected = expect_0[..num_out].to_vec();
    expected.extend(expect_1[..num_out].iter().copied());

    // TODO: Eventually add approx comparison that can deal with arbitrary floats. Manually
    // double check for now
    let actual_2 = client.read_one_unchecked(handle1);
    let actual_2 = F::from_bytes(&actual_2);
    println!("actual_2: {actual_2:?}");

    // Data rounded to the nearest e2m3 value
    let expected_data = as_type![F: -2.0, 1.75, 0.375, 1.25];

    assert_eq!(actual, &expected);
    assert_eq!(&actual_2[..num_out], &expected_data[..num_out]);
}

#[allow(clippy::unusual_byte_groupings, reason = "Split by float components")]
pub fn test_fp4<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    vector_size: VectorSize,
) {
    if !e2m1x2::supported_uses(&client).contains(TypeUsage::Conversion) {
        println!("Unsupported, skipping");
        return;
    }

    let data = as_type![F: -2.1, 1.8, 0.4, 1.2];
    let num_out = vector_size;
    let handle1 = client.create_from_slice(F::as_bytes(&data[..num_out]));
    let handle2 = client.empty(num_out / 2 * size_of::<u8>());

    unsafe {
        kernel_fp4::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            vector_size,
            vector_size / 2,
            BufferArg::from_raw_parts(handle1.clone(), num_out),
            BufferArg::from_raw_parts(handle2.clone(), 2 * num_out),
        )
    };

    let actual = client.read_one_unchecked(handle2);
    let actual = u8::from_bytes(&actual);
    // LITTLE ENDIAN FOR PACKED VALUES
    let expect_0: Vec<u8> = vec![0b0_10_0__1_10_0, 0b0_01_0__0_00_1];
    let expected = expect_0[..num_out / 2].to_vec();

    let actual_2 = client.read_one_unchecked(handle1);
    let actual_2 = F::from_bytes(&actual_2);
    println!("actual_2: {actual_2:?}");

    // Data rounded to the nearest e2m1 value
    let expected_data = as_type![F: -2.0, 2.0, 0.5, 1.0];

    assert_eq!(actual, &expected);
    assert_eq!(&actual_2[..num_out], &expected_data[..num_out]);
}

fn fp8_supported<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let usable = |uses: EnumSet<TypeUsage>| uses.contains(TypeUsage::Conversion);
    usable(e4m3::supported_uses(client)) && usable(e5m2::supported_uses(client))
}

macro_rules! assert_encoded {
    ($fmt:ident, $actual:expr, $value:expr) => {{
        let actual = $actual;
        let value: f32 = $value;
        if value.is_nan() {
            assert!(
                $fmt::from_bits(actual).is_nan(),
                "{} of NaN: {actual:#04x}",
                stringify!($fmt)
            );
        } else {
            assert_eq!(
                actual,
                $fmt::from_f32(value).to_bits(),
                "{} of {value:e} ({:#010x})",
                stringify!($fmt),
                value.to_bits()
            );
        }
    }};
}

/// Each fp8 test is the same test once per format, so both the kernels and the bodies are
/// generated. The format stays a concrete type: `e4m3` and `e5m2` are `Scalar + CubePrimitive`,
/// not `Float`, so there is no bound to be generic over.
macro_rules! fp8_format_tests {
    ($fmt:ident) => {
        ::paste::paste! {
            pub mod [< fp8_ $fmt >] {
                use super::*;

                #[cube(launch_unchecked)]
                pub fn kernel_decode<N: Size>(
                    input: &[Vector<u8, N>],
                    out: &mut [Vector<f32, N>],
                ) {
                    if ABSOLUTE_POS < input.len() {
                        out[ABSOLUTE_POS] =
                            Vector::cast_from(Vector::<$fmt, N>::reinterpret(input[ABSOLUTE_POS]));
                    }
                }

                #[cube(launch_unchecked)]
                pub fn kernel_encode<N: Size>(
                    input: &[Vector<f32, N>],
                    out: &mut [Vector<u8, N>],
                ) {
                    if ABSOLUTE_POS < input.len() {
                        out[ABSOLUTE_POS] =
                            Vector::reinterpret(Vector::<$fmt, N>::cast_from(input[ABSOLUTE_POS]));
                    }
                }

                /// The input is derived from the position rather than read, so sweeping every
                /// `f32` uploads no operands at all.
                #[cube(launch_unchecked)]
                pub fn kernel_sweep(base: u32, out: &mut [u8]) {
                    if ABSOLUTE_POS < out.len() {
                        let value = f32::reinterpret(base + ABSOLUTE_POS as u32);
                        out[ABSOLUTE_POS] = u8::reinterpret($fmt::cast_from(value));
                    }
                }

                pub fn decode_exhaustive<R: Runtime>(
                    client: ComputeClient<R>,
                    vector_size: VectorSize,
                ) {
                    if !fp8_supported(&client) {
                        println!("Unsupported, skipping");
                        return;
                    }

                    let bytes: Vec<u8> = (0..=u8::MAX).collect();
                    let input = client.create_from_slice(u8::as_bytes(&bytes));
                    let out = client.empty(bytes.len() * size_of::<f32>());
                    let vectors = bytes.len() / vector_size;

                    unsafe {
                        kernel_decode::launch_unchecked::<R>(
                            &client,
                            CubeCount::Static(vectors.div_ceil(32) as u32, 1, 1),
                            CubeDim::new_1d(32),
                            vector_size,
                            BufferArg::from_raw_parts(input, bytes.len()),
                            BufferArg::from_raw_parts(out.clone(), bytes.len()),
                        )
                    };

                    let actual = client.read_one_unchecked(out);
                    for (bits, actual) in bytes.iter().zip(f32::from_bytes(&actual)) {
                        assert_same_float(
                            *actual,
                            $fmt::from_bits(*bits).to_f32(),
                            stringify!($fmt),
                            *bits as u32,
                        );
                    }
                }

                pub fn encode_exhaustive<R: Runtime>(
                    client: ComputeClient<R>,
                    vector_size: VectorSize,
                ) {
                    if !fp8_supported(&client) {
                        println!("Unsupported, skipping");
                        return;
                    }

                    let values = fp8_encode_inputs(vector_size);
                    let input = client.create_from_slice(f32::as_bytes(&values));
                    let out = client.empty(values.len());
                    let vectors = values.len() / vector_size;

                    unsafe {
                        kernel_encode::launch_unchecked::<R>(
                            &client,
                            CubeCount::Static(vectors.div_ceil(64) as u32, 1, 1),
                            CubeDim::new_1d(64),
                            vector_size,
                            BufferArg::from_raw_parts(input, values.len()),
                            BufferArg::from_raw_parts(out.clone(), values.len()),
                        )
                    };

                    let actual = client.read_one_unchecked(out);
                    for (value, actual) in values.iter().zip(u8::from_bytes(&actual)) {
                        assert_encoded!($fmt, *actual, *value);
                    }
                }

                /// Every `f32` bit pattern, in chunks, against the host codec. Slow by
                /// construction, so it is `#[ignore]`d rather than gated: an ignored test still
                /// compiles, so it cannot rot unnoticed.
                pub fn encode_sweep<R: Runtime>(client: ComputeClient<R>) {
                    if !fp8_supported(&client) {
                        println!("Unsupported, skipping");
                        return;
                    }

                    // Sized so the cube count stays under the 65535 per dimension every backend
                    // allows.
                    const CUBE_DIM: usize = 256;
                    const CHUNK: usize = 1 << 22;
                    let out = client.empty(CHUNK);

                    for base in (0..=(u32::MAX as u64)).step_by(CHUNK) {
                        let base = base as u32;
                        unsafe {
                            kernel_sweep::launch_unchecked::<R>(
                                &client,
                                CubeCount::Static((CHUNK / CUBE_DIM) as u32, 1, 1),
                                CubeDim::new_1d(CUBE_DIM as u32),
                                base,
                                BufferArg::from_raw_parts(out.clone(), CHUNK),
                            )
                        };

                        let actual = client.read_one_unchecked(out.clone());
                        for (offset, actual) in u8::from_bytes(&actual).iter().enumerate() {
                            assert_encoded!($fmt, *actual, f32::from_bits(base + offset as u32));
                        }
                    }
                }
            }
        }
    };
}

fp8_format_tests!(e4m3);
fp8_format_tests!(e5m2);

/// Every `f16`, so each exponent and every mantissa tie an `f16` can name is covered, plus the
/// boundaries only an `f32` can express. Padded to the vector size the kernel launches at.
fn fp8_encode_inputs(vector_size: VectorSize) -> Vec<f32> {
    let mut values: Vec<f32> = (0..=u16::MAX)
        .map(|bits| half::f16::from_bits(bits).to_f32())
        .collect();
    values.extend(fp8_encode_edges());
    while !values.len().is_multiple_of(vector_size) {
        values.push(0.0);
    }
    values
}

/// Where rounding decisions actually happen, which is nowhere near uniformly spread. Per format:
/// the largest finite value, the tie above it that decides saturation, and the step past it; the
/// smallest normal and the subnormal step, walked at the exact, half, quarter and three-quarter
/// points so round-to-nearest-even is forced both ways; the subnormal-to-normal boundary from
/// either side; and mantissa ties at several exponents together with their neighbouring ULPs.
fn fp8_encode_edges() -> Vec<f32> {
    let mut edges = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        448.0,
        -448.0,
        464.0,
        480.0,
        57344.0,
        -57344.0,
        61440.0,
        65536.0,
    ];
    for (min_normal, step) in [(1.0 / 64.0, 1.0 / 512.0), (1.0 / 16384.0, 1.0 / 65536.0)] {
        for k in 0..8 {
            let below = k as f32 * step;
            edges.extend([
                below,
                below + step / 2.0,
                below + step / 4.0,
                below + step * 3.0 / 4.0,
            ]);
        }
        edges.extend([
            step / 2.0,
            step / 4.0,
            min_normal,
            min_normal * 0.999,
            min_normal * 1.001,
        ]);
    }
    for value in [1.0f32, 1.5, 3.0, 100.0, 1000.0] {
        for step in [1.0 / 8.0, 1.0 / 4.0] {
            let half = value * step / 2.0;
            edges.extend([value + half, value + half * 3.0, value - half]);
            edges.extend([
                f32::from_bits((value + half).to_bits() + 1),
                f32::from_bits((value + half).to_bits() - 1),
            ]);
        }
    }
    let negatives: Vec<f32> = edges.iter().map(|value| -value).collect();
    edges.extend(negatives);
    edges
}

fn assert_same_float(actual: f32, expected: f32, format: &str, bits: u32) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{format} {bits:#04x}: expected NaN, got {actual:e}"
        );
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{format} {bits:#04x}: expected {expected:e}, got {actual:e}"
        );
    }
}

pub fn test_scale<R: Runtime>(client: ComputeClient<R>, vector_size: VectorSize) {
    if !ue8m0::supported_uses(&client).contains(TypeUsage::Conversion) {
        println!("Unsupported, skipping");
        return;
    }

    let data = [2.0, 1024.0, 57312.0, f32::from_bits(0x7F000000)];
    let num_out = vector_size;
    let handle1 = client.create_from_slice(f32::as_bytes(&data[..num_out]));
    let handle2 = client.empty(num_out * size_of::<u8>());

    unsafe {
        kernel_scale::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            vector_size,
            BufferArg::from_raw_parts(handle1.clone(), num_out),
            BufferArg::from_raw_parts(handle2.clone(), num_out),
        )
    };

    let actual = client.read_one_unchecked(handle2);
    let actual = u8::from_bytes(&actual);
    let expect: Vec<u8> = vec![0b1000_0000, 0b1000_1001, 0b1000_1111, 0b1111_1110];

    // TODO: Eventually add approx comparison that can deal with arbitrary floats. Manually
    // double check for now
    let actual_2 = client.read_one_unchecked(handle1);
    let actual_2 = f32::from_bytes(&actual_2);
    println!("actual_2: {actual_2:?}");

    assert_eq!(actual, &expect[..num_out]);
    //assert_eq!(&actual_2[..num_out], &data[..num_out]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_minifloat {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_fp8() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::minifloat::test_fp8::<TestRuntime, FloatType>(
                client.clone(),
                1,
            );
            cubecl_core::runtime_tests::minifloat::test_fp8::<TestRuntime, FloatType>(
                client.clone(),
                2,
            );
            cubecl_core::runtime_tests::minifloat::test_fp8::<TestRuntime, FloatType>(
                client.clone(),
                4,
            );
        }

        mod e4m3 {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn decode_exhaustive() {
                let client = TestRuntime::client(&Default::default());
                for vector_size in [1, 4] {
                    cubecl_core::runtime_tests::minifloat::fp8_e4m3::decode_exhaustive::<
                        TestRuntime,
                    >(client.clone(), vector_size);
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn encode_exhaustive() {
                let client = TestRuntime::client(&Default::default());
                for vector_size in [1, 4] {
                    cubecl_core::runtime_tests::minifloat::fp8_e4m3::encode_exhaustive::<
                        TestRuntime,
                    >(client.clone(), vector_size);
                }
            }

            #[$crate::runtime_tests::test_log::test]
            #[ignore = "sweeps every f32; run with --ignored"]
            fn encode_sweep() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::minifloat::fp8_e4m3::encode_sweep::<TestRuntime>(
                    client,
                );
            }
        }

        mod e5m2 {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn decode_exhaustive() {
                let client = TestRuntime::client(&Default::default());
                for vector_size in [1, 4] {
                    cubecl_core::runtime_tests::minifloat::fp8_e5m2::decode_exhaustive::<
                        TestRuntime,
                    >(client.clone(), vector_size);
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn encode_exhaustive() {
                let client = TestRuntime::client(&Default::default());
                for vector_size in [1, 4] {
                    cubecl_core::runtime_tests::minifloat::fp8_e5m2::encode_exhaustive::<
                        TestRuntime,
                    >(client.clone(), vector_size);
                }
            }

            #[$crate::runtime_tests::test_log::test]
            #[ignore = "sweeps every f32; run with --ignored"]
            fn encode_sweep() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::minifloat::fp8_e5m2::encode_sweep::<TestRuntime>(
                    client,
                );
            }
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_fp6() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::minifloat::test_fp6::<TestRuntime, FloatType>(
                client.clone(),
                1,
            );
            cubecl_core::runtime_tests::minifloat::test_fp6::<TestRuntime, FloatType>(
                client.clone(),
                2,
            );
            cubecl_core::runtime_tests::minifloat::test_fp6::<TestRuntime, FloatType>(
                client.clone(),
                4,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_fp4() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::minifloat::test_fp4::<TestRuntime, FloatType>(
                client.clone(),
                2,
            );
            cubecl_core::runtime_tests::minifloat::test_fp4::<TestRuntime, FloatType>(
                client.clone(),
                4,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_scale() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(client.clone(), 1);
            cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(client.clone(), 2);
            cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(client.clone(), 4);
        }
    };
}
