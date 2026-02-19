use crate::{self as cubecl};
use alloc::vec;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_saturating_add<I: Int>(
    lhs: &Array<Line<I>>,
    rhs: &Array<Line<I>>,
    output: &mut Array<Line<I>>,
) {
    if (UNIT_POS as usize) < output.len() {
        output[UNIT_POS as usize] =
            Line::<I>::saturating_add(lhs[UNIT_POS as usize], rhs[UNIT_POS as usize]);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_saturating_sub<I: Int>(
    lhs: &Array<Line<I>>,
    rhs: &Array<Line<I>>,
    output: &mut Array<Line<I>>,
) {
    if (UNIT_POS as usize) < output.len() {
        output[UNIT_POS as usize] =
            Line::<I>::saturating_sub(lhs[UNIT_POS as usize], rhs[UNIT_POS as usize]);
    }
}

#[allow(clippy::needless_range_loop)]
pub fn test_saturating_add_unsigned<R: Runtime, I: Int + CubeElement>(
    client: ComputeClient<R>,
    line_size: LineSize,
) {
    if I::cube_type() == u64::cube_type() {
        // Seems to have inexplicable crash on Vulkan with no validation errors. Likely a driver
        // bug, recheck if you see this and the Nvidia driver is > 591.59.
        return;
    }

    let lhs = vec![
        I::new(2),
        I::max_value(),
        I::max_value() - I::new(10),
        I::new(20),
    ];
    let rhs = vec![I::new(10), I::new(1), I::new(9), I::max_value()];
    let out = vec![
        I::new(12),
        I::max_value(),
        I::max_value() - I::new(1),
        I::max_value(),
    ];

    let lhs_handle = client.create_from_slice(I::as_bytes(&lhs));
    let rhs_handle = client.create_from_slice(I::as_bytes(&rhs));
    let out_handle = client.empty(4 * size_of::<I>());

    unsafe {
        kernel_saturating_add::launch_unchecked::<I, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(out.len() as u32),
            ArrayArg::from_raw_parts::<I>(&lhs_handle, 4, line_size),
            ArrayArg::from_raw_parts::<I>(&rhs_handle, 4, line_size),
            ArrayArg::from_raw_parts::<I>(&out_handle, 4, line_size),
        )
    }
    let actual = client.read_one_unchecked(out_handle);
    let actual = I::from_bytes(&actual);

    assert_eq!(actual, out);
}

#[allow(clippy::needless_range_loop)]
pub fn test_saturating_sub_unsigned<R: Runtime, I: Int + CubeElement>(
    client: ComputeClient<R>,
    line_size: LineSize,
) {
    if I::cube_type() == u64::cube_type() {
        // Seems to have inexplicable crash on Vulkan with no validation errors. Likely a driver
        // bug, recheck if you see this and the Nvidia driver is > 591.59.
        return;
    }

    let lhs = vec![
        I::new(2),
        I::new(4),
        I::new(10),
        I::max_value() - I::new(10),
    ];
    let rhs = vec![I::new(1), I::new(6), I::new(8), I::max_value()];
    let out = vec![I::new(1), I::new(0), I::new(2), I::new(0)];

    let lhs_handle = client.create_from_slice(I::as_bytes(&lhs));
    let rhs_handle = client.create_from_slice(I::as_bytes(&rhs));
    let out_handle = client.empty(4 * size_of::<I>());

    unsafe {
        kernel_saturating_sub::launch_unchecked::<I, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(out.len() as u32),
            ArrayArg::from_raw_parts::<I>(&lhs_handle, 4, line_size),
            ArrayArg::from_raw_parts::<I>(&rhs_handle, 4, line_size),
            ArrayArg::from_raw_parts::<I>(&out_handle, 4, line_size),
        )
    }
    let actual = client.read_one_unchecked(out_handle);
    let actual = I::from_bytes(&actual);

    assert_eq!(actual, out);
}

// Signed has a lot more possible cases due to overflow/underflow
#[allow(clippy::needless_range_loop)]
pub fn test_saturating_add_signed<R: Runtime, I: Int + CubeElement>(
    client: ComputeClient<R>,
    line_size: LineSize,
) {
    let lhs = vec![
        I::new(0),
        I::new(0),
        I::new(0),
        I::new(5),
        I::new(-5),
        I::new(10),
        I::new(-10),
        I::new(50),
        I::new(30),
        I::new(10),
        I::max_value(),
        I::new(1),
        I::min_value(),
        I::new(-1),
        I::max_value() - I::new(1),
        I::min_value() + I::new(1),
    ];
    let rhs = vec![
        I::new(0),
        I::new(5),
        I::new(-5),
        I::new(0),
        I::new(0),
        I::new(20),
        I::new(-20),
        I::new(-30),
        I::new(-50),
        I::new(-10),
        I::new(1),
        I::max_value(),
        I::new(-1),
        I::min_value(),
        I::new(1),
        I::new(-1),
    ];
    let out = vec![
        I::new(0),
        I::new(5),
        I::new(-5),
        I::new(5),
        I::new(-5),
        I::new(30),
        I::new(-30),
        I::new(20),
        I::new(-20),
        I::new(0),
        I::max_value(),
        I::max_value(),
        I::min_value(),
        I::min_value(),
        I::max_value(),
        I::min_value(),
    ];

    let lhs_handle = client.create_from_slice(I::as_bytes(&lhs));
    let rhs_handle = client.create_from_slice(I::as_bytes(&rhs));
    let out_handle = client.empty(16 * size_of::<I>());

    unsafe {
        kernel_saturating_add::launch_unchecked::<I, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(out.len() as u32),
            ArrayArg::from_raw_parts::<I>(&lhs_handle, 16, line_size),
            ArrayArg::from_raw_parts::<I>(&rhs_handle, 16, line_size),
            ArrayArg::from_raw_parts::<I>(&out_handle, 16, line_size),
        )
    }
    let actual = client.read_one_unchecked(out_handle);
    let actual = I::from_bytes(&actual);

    assert_eq!(actual, out);
}

// Signed has a lot more possible cases due to overflow/underflow
#[allow(clippy::needless_range_loop)]
pub fn test_saturating_sub_signed<R: Runtime, I: Int + CubeElement>(
    client: ComputeClient<R>,
    line_size: LineSize,
) {
    let lhs = vec![
        I::new(0),                  // 1. Zero identity
        I::new(0),                  // 2. Subtract positive from zero
        I::new(0),                  // 3. Subtract negative from zero
        I::new(10),                 // 4. Normal positive subtraction
        I::new(-10),                // 5. Normal negative subtraction
        I::new(20),                 // 6. Positive - positive (positive result)
        I::new(5),                  // 7. Positive - positive (negative result)
        I::new(-5),                 // 8. Negative - negative (positive result)
        I::new(-20),                // 9. Negative - negative (negative result)
        I::max_value(),             // 10. Max - negative (would overflow)
        I::max_value(),             // 11. Max - positive (normal)
        I::min_value(),             // 12. Min - positive (would underflow)
        I::min_value(),             // 13. Min - negative (normal)
        I::max_value() - I::new(1), // 14. Near max - negative
        I::min_value() + I::new(1), // 15. Near min - positive
        I::new(50),                 // 16. Normal mixed signs
    ];
    let rhs = vec![
        I::new(0),
        I::new(5),
        I::new(-5),
        I::new(3),
        I::new(-3),
        I::new(15),
        I::new(10),
        I::new(-10),
        I::new(-5),
        I::new(-1),
        I::new(1),
        I::new(1),
        I::new(-1),
        I::new(-1),
        I::new(1),
        I::new(-30),
    ];
    let out = vec![
        I::new(0),
        I::new(-5),
        I::new(5),
        I::new(7),
        I::new(-7),
        I::new(5),
        I::new(-5),
        I::new(5),
        I::new(-15),
        I::max_value(), // Saturates at max
        I::max_value() - I::new(1),
        I::min_value(), // Saturates at min
        I::min_value() + I::new(1),
        I::max_value(), // Would overflow: (MAX-1) - (-1) = MAX
        I::min_value(), // Would underflow: (MIN+1) - 1 = MIN
        I::new(80),     // 50 - (-30) = 80
    ];

    let lhs_handle = client.create_from_slice(I::as_bytes(&lhs));
    let rhs_handle = client.create_from_slice(I::as_bytes(&rhs));
    let out_handle = client.empty(16 * size_of::<I>());

    unsafe {
        kernel_saturating_sub::launch_unchecked::<I, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(out.len() as u32),
            ArrayArg::from_raw_parts::<I>(&lhs_handle, 16, line_size),
            ArrayArg::from_raw_parts::<I>(&rhs_handle, 16, line_size),
            ArrayArg::from_raw_parts::<I>(&out_handle, 16, line_size),
        )
    }
    let actual = client.read_one_unchecked(out_handle);
    let actual = I::from_bytes(&actual);

    assert_eq!(actual, out);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_saturating_uint {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_saturating_add_unsigned() {
            let client = TestRuntime::client(&Default::default());
            let test = cubecl_core::runtime_tests::saturating::test_saturating_add_unsigned::<
                TestRuntime,
                UintType,
            >;
            test(client.clone(), 1);
            test(client.clone(), 2);
            test(client, 4);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_saturating_sub_unsigned() {
            let client = TestRuntime::client(&Default::default());
            let test = cubecl_core::runtime_tests::saturating::test_saturating_sub_unsigned::<
                TestRuntime,
                UintType,
            >;
            test(client.clone(), 1);
            test(client.clone(), 2);
            test(client, 4);
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_saturating_int {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_saturating_add_signed() {
            let client = TestRuntime::client(&Default::default());
            let test = cubecl_core::runtime_tests::saturating::test_saturating_add_signed::<
                TestRuntime,
                IntType,
            >;
            test(client.clone(), 1);
            test(client.clone(), 2);
            test(client, 4);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_saturating_sub_signed() {
            let client = TestRuntime::client(&Default::default());
            let test = cubecl_core::runtime_tests::saturating::test_saturating_sub_signed::<
                TestRuntime,
                IntType,
            >;
            test(client.clone(), 1);
            test(client.clone(), 2);
            test(client, 4);
        }
    };
}
