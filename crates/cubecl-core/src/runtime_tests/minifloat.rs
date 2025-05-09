use crate::{self as cubecl, as_type};

use cubecl::prelude::*;
use cubecl_common::{e2m3, e3m2, e4m3, e5m2, ue8m0};

#[cube(launch_unchecked)]
pub fn kernel_fp8<F: Float>(input: &mut Array<Line<F>>, out: &mut Array<Line<u8>>) {
    if ABSOLUTE_POS == 0 {
        let value = input[0];

        out[0] = Line::reinterpret(Line::<e4m3>::cast_from(value));
        out[1] = Line::reinterpret(Line::<e5m2>::cast_from(value));
        input[0] = Line::cast_from(Line::<e4m3>::reinterpret(out[0]));
    }
}

#[cube(launch_unchecked)]
pub fn kernel_fp6<F: Float>(input: &mut Array<Line<F>>, out: &mut Array<Line<u8>>) {
    if ABSOLUTE_POS == 0 {
        let value = input[0];

        out[0] = Line::reinterpret(Line::<e2m3>::cast_from(value));
        out[1] = Line::reinterpret(Line::<e3m2>::cast_from(value));
        input[0] = Line::cast_from(Line::<e2m3>::reinterpret(out[0]));
    }
}

#[cube(launch_unchecked)]
pub fn kernel_scale(input: &mut Array<Line<f32>>, out: &mut Array<Line<ue8m0>>) {
    if ABSOLUTE_POS == 0 {
        let value = input[0];

        out[0] = Line::<ue8m0>::cast_from(value);
        input[0] = Line::cast_from(out[0]);
    }
}

#[allow(clippy::unusual_byte_groupings, reason = "Split by float components")]
pub fn test_fp8<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    vectorization: u8,
) {
    if !e4m3::is_supported(&client) {
        println!("Unsupported, skipping");
        return;
    }

    let data = as_type![F: -2.1, 1.8, 0.4, 1.2];
    let num_out = vectorization as usize;
    let handle1 = client.create(F::as_bytes(&data[..num_out]));
    let handle2 = client.empty(2 * num_out * size_of::<u8>());

    unsafe {
        kernel_fp8::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&handle1, num_out, vectorization),
            ArrayArg::from_raw_parts::<u8>(&handle2, 2 * num_out, vectorization),
        )
    };

    let actual = client.read_one(handle2.binding());
    let actual = u8::from_bytes(&actual);
    let expect_0: Vec<u8> = vec![0b1_1000_000, 0b0_0111_110, 0b0_0101_101, 0b0_0111_010];
    let expect_1: Vec<u8> = vec![0b1_10000_00, 0b0_01111_11, 0b0_01101_10, 0b0_01111_01];
    let mut expected = expect_0[..num_out].to_vec();
    expected.extend(expect_1[..num_out].iter().copied());

    // TODO: Eventually add approx comparison that can deal with arbitrary floats. Manually
    // double check for now
    let actual_2 = client.read_one(handle1.binding());
    let actual_2 = F::from_bytes(&actual_2);
    println!("actual_2: {actual_2:?}");

    assert_eq!(actual, &expected);
    //assert_eq!(&actual_2[..num_out], &data[..num_out]);
}

#[allow(clippy::unusual_byte_groupings, reason = "Split by float components")]
pub fn test_fp6<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    vectorization: u8,
) {
    if !e2m3::is_supported(&client) {
        println!("Unsupported, skipping");
        return;
    }

    let data = as_type![F: -2.1, 1.8, 0.4, 1.2];
    let num_out = vectorization as usize;
    let handle1 = client.create(F::as_bytes(&data[..num_out]));
    let handle2 = client.empty(2 * num_out * size_of::<u8>());

    unsafe {
        kernel_fp6::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&handle1, num_out, vectorization),
            ArrayArg::from_raw_parts::<u8>(&handle2, 2 * num_out, vectorization),
        )
    };

    let actual = client.read_one(handle2.binding());
    let actual = u8::from_bytes(&actual);
    let expect_0: Vec<u8> = vec![0b1_10_000, 0b0_01_110, 0b0_00_011, 0b0_01_010];
    let expect_1: Vec<u8> = vec![0b1_100_00, 0b0_011_11, 0b0_001_10, 0b0_011_01];
    let mut expected = expect_0[..num_out].to_vec();
    expected.extend(expect_1[..num_out].iter().copied());

    // TODO: Eventually add approx comparison that can deal with arbitrary floats. Manually
    // double check for now
    let actual_2 = client.read_one(handle1.binding());
    let actual_2 = F::from_bytes(&actual_2);
    println!("actual_2: {actual_2:?}");

    assert_eq!(actual, &expected);
    //assert_eq!(&actual_2[..num_out], &data[..num_out]);
}

pub fn test_scale<R: Runtime>(client: ComputeClient<R::Server, R::Channel>, vectorization: u8) {
    if !ue8m0::is_supported(&client) {
        println!("Unsupported, skipping");
        return;
    }

    let data = [2.0, 1024.0, 57312.0, f32::from_bits(0x7F000000)];
    let num_out = vectorization as usize;
    let handle1 = client.create(f32::as_bytes(&data[..num_out]));
    let handle2 = client.empty(num_out * size_of::<u8>());

    unsafe {
        kernel_scale::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&handle1, num_out, vectorization),
            ArrayArg::from_raw_parts::<u8>(&handle2, num_out, vectorization),
        )
    };

    let actual = client.read_one(handle2.binding());
    let actual = u8::from_bytes(&actual);
    let expect: Vec<u8> = vec![0b1000_0000, 0b1000_1001, 0b1000_1111, 0b1111_1110];

    // TODO: Eventually add approx comparison that can deal with arbitrary floats. Manually
    // double check for now
    let actual_2 = client.read_one(handle1.binding());
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

        #[test]
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

        #[test]
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

        // #[test]
        // fn test_fp4() {
        //     let client = TestRuntime::client(&Default::default());
        //     cubecl_core::runtime_tests::minifloat::test_fp4::<TestRuntime>(client.clone(), 2);
        //     cubecl_core::runtime_tests::minifloat::test_fp4::<TestRuntime>(client.clone(), 4);
        // }

        #[test]
        fn test_scale() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(client.clone(), 1);
            cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(client.clone(), 2);
            cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(client.clone(), 4);
        }
    };
}
