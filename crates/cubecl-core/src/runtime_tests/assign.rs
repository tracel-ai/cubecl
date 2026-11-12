use crate as cubecl;
use cubecl_runtime::runtime::Runtime;

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_assign<F: Float>(output: &mut [F]) {
    if UNIT_POS == 0 {
        let item = F::new(5.0);
        output[0] = item;
    }
}

#[cube(launch)]
pub fn kernel_assign_one_tuple<F: Float>(output: &mut [F]) {
    if UNIT_POS == 0 {
        let item = (F::new(5.0),);
        output[0] = item.0;
    }
}

#[cube(launch)]
pub fn kernel_add_assign_array<F: Float, N: Size>(output: &mut [Vector<F, N>]) {
    if UNIT_POS == 0 {
        output[0] = Vector::new(F::new(5.0));
        output[0] += Vector::new(F::new(1.0));
    }
}

#[cube(launch)]
pub fn kernel_add_assign_vector<F: Float, N: Size>(output: &mut [Vector<F, N>]) {
    let mut vector = Vector::new(F::new(1.0));

    if UNIT_POS == 0 {
        #[unroll]
        for i in 0..N::value() {
            // This is awkward but shouldn't be done anyways in a real program
            vector.insert(i, vector.extract(i) + F::cast_from(i));
        }
        output[0] = vector;
    }
}

#[cube(launch)]
pub fn kernel_assign_ref<F: Float>(output: &mut [F]) {
    if UNIT_POS == 0 {
        let mut value = F::new(1.0);
        assign_ref::<F>(&mut value);
        output[0] = value;
    }
}

#[cube]
fn assign_ref<F: Float>(value: &mut F) {
    *value = F::new(5.0);
}

#[cube(launch)]
fn assign_mut_expression(output: &mut [u32]) {
    let thread_idx = UNIT_POS as usize;
    let num_threads = CUBE_DIM as usize;
    let mut step = (256 / 2usize).runtime();

    while step > 0 {
        for pos in range_stepped(thread_idx, step, num_threads) {
            output[pos] = 0;
        }
        step /= 2;
    }
}

/// A comptime initialiser made runtime takes a runtime value in a comptime branch.
#[cube(launch)]
fn kernel_compound_assign_runtime_word(
    perm: &[u32],
    output: &mut [u32],
    #[comptime] cycle: u32,
    #[comptime] has_perm: bool,
) {
    let mut word = cycle.runtime();
    if comptime!(has_perm) {
        word ^= perm[0];
    }
    output[0] = word;
}

/// Every compound assignment onto a runtime binding, with comptime and runtime values.
#[cube(launch)]
fn kernel_compound_assign_kinds(input: &[u32], output: &mut [u32], #[comptime] step: u32) {
    let mut offset = step.runtime();
    offset += step;
    offset *= 3;
    offset -= 1;
    offset /= 2;
    offset %= 4;
    offset |= 8;
    offset &= 12;
    offset ^= 1;
    offset <<= 2;
    offset >>= 1;
    output[0] = offset;

    let mut value = input[0];
    value += step;
    value += 1;
    value ^= input[1];
    value <<= offset;
    output[1] = value;
}

pub fn test_compound_assign_runtime_word<R: Runtime>(client: Client) {
    for (has_perm, expected) in [(true, 0b1010 ^ 0b0110), (false, 0b1010)] {
        let perm = client.create_from_slice(u32::as_bytes(&[0b0110]));
        let output = client.create_from_slice(u32::as_bytes(&[0]));

        kernel_compound_assign_runtime_word::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(perm, 1) },
            unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
            0b1010,
            has_perm,
        );

        let actual = client.read_one(output).unwrap();
        let actual = u32::from_bytes(&actual);
        assert_eq!(actual, [expected], "has_perm = {has_perm}");
    }
}

pub fn test_compound_assign_kinds<R: Runtime>(client: Client) {
    let step = 5u32;
    let mut offset = step;
    offset += step;
    offset *= 3;
    offset -= 1;
    offset /= 2;
    offset %= 4;
    offset |= 8;
    offset &= 12;
    offset ^= 1;
    offset <<= 2;
    offset >>= 1;
    let value = ((7 + step + 1) ^ 3) << offset;

    let input = client.create_from_slice(u32::as_bytes(&[7, 3]));
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    kernel_compound_assign_kinds::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
        step,
    );

    let actual = client.read_one(output).unwrap();
    let actual = u32::from_bytes(&actual);
    assert_eq!(actual, [offset, value]);
}

pub fn test_kernel_assign_scalar<R: Runtime, F: Float + CubeElement>(client: Client) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    kernel_assign::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_kernel_assign_one_tuple<R: Runtime, F: Float + CubeElement>(client: Client) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0)]));

    kernel_assign_one_tuple::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_kernel_add_assign_array<R: Runtime, F: Float + CubeElement>(client: Client) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    let vectorization = 2;

    kernel_add_assign_array::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        vectorization,
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(6.0));
}

pub fn test_kernel_add_assign_vector<R: Runtime, F: Float + CubeElement>(client: Client) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    let vectorization = 2;

    kernel_add_assign_vector::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        vectorization,
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1.0));
    assert_eq!(actual[1], F::new(2.0));
}

pub fn test_kernel_assign_ref<R: Runtime, F: Float + CubeElement>(client: Client) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    kernel_assign_ref::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_assign_mut_expr<R: Runtime>(client: Client) {
    let output = client.empty(128 * size_of::<u32>());

    unsafe {
        assign_mut_expression::launch(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 128),
        );
    }

    client.read_one(output).unwrap();
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_assign {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_assign_scalar() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_assign_scalar::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_assign_one_tuple() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_assign_one_tuple::<
                TestRuntime,
                FloatType,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_add_assign_array() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_add_assign_array::<
                TestRuntime,
                FloatType,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_add_assign_vector() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_add_assign_vector::<
                TestRuntime,
                FloatType,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_assign_ref() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_assign_ref::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_compound_assign_runtime_word() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_compound_assign_runtime_word::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_compound_assign_kinds() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_compound_assign_kinds::<TestRuntime>(client);
        }

        #[ignore = "Broken right now, should be fixed with future frontend work"]
        #[$crate::runtime_tests::test_log::test]
        fn test_assign_mut_expr() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_assign_mut_expr::<TestRuntime>(client);
        }
    };
}
