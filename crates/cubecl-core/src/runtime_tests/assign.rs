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
    if has_perm {
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

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Cursor {
    slot: u32,
}

#[cube]
fn take_slot(cursor: &mut Cursor) -> u32 {
    let start = cursor.slot;
    cursor.slot += 3u32;
    start
}

#[cube(launch_unchecked)]
fn kernel_copy_native_struct_field(output: &mut [u32]) {
    let mut cursor = Cursor {
        slot: 5u32.runtime(),
    };
    let start = take_slot(&mut cursor);
    output[0] = start;
    output[1] = cursor.slot;
}

#[derive(Clone, Copy)]
struct HostSlot(u32);

#[derive(CubeType, Clone, Copy)]
struct ComptimeCursor {
    #[cube(comptime)]
    slot: HostSlot,
}

#[cube]
fn take_comptime_slot(cursor: &mut ComptimeCursor) -> u32 {
    let start = cursor.slot;
    comptime!(cursor.slot = HostSlot(8));
    comptime!(start.0).runtime()
}

#[cube(launch_unchecked)]
fn kernel_copy_comptime_struct_field(output: &mut [u32]) {
    let mut cursor = ComptimeCursor {
        slot: comptime!(HostSlot(5)),
    };
    let start = take_comptime_slot(&mut cursor);
    output[0] = start;
    output[1] = comptime!(cursor.slot.0).runtime();
}

pub fn test_copy_native_struct_field<R: Runtime>(client: Client) {
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    unsafe {
        kernel_copy_native_struct_field::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let actual = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&actual), [5, 8]);
}

pub fn test_copy_comptime_struct_field<R: Runtime>(client: Client) {
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    unsafe {
        kernel_copy_comptime_struct_field::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let actual = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&actual), [5, 8]);
}

#[derive(CubeType)]
struct RefCursor<'a> {
    slot: &'a mut u32,
}

#[cube]
fn write_slot_through_field<'a>(value: &'a mut u32) {
    let cursor = RefCursor::<'a> { slot: value };
    let slot = cursor.slot;
    *slot = 8u32;
}

#[cube(launch_unchecked)]
fn kernel_mut_ref_struct_field(output: &mut [u32]) {
    let mut value = 5u32.runtime();
    output[0] = value;
    write_slot_through_field(&mut value);
    output[1] = value;
}

pub fn test_mut_ref_struct_field<R: Runtime>(client: Client) {
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    unsafe {
        kernel_mut_ref_struct_field::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let actual = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&actual), [5, 8]);
}

#[cube]
impl Cursor {
    fn snapshot_and_bump(self, original: &mut Cursor) -> u32 {
        original.slot += 3u32;
        self.slot
    }
}

#[cube(launch_unchecked)]
fn kernel_copy_owned_receiver(output: &mut [u32]) {
    let mut cursor = Cursor {
        slot: 5u32.runtime(),
    };
    let snapshot = cursor.snapshot_and_bump(&mut cursor);
    output[0] = snapshot;
    output[1] = cursor.slot;
}

pub fn test_copy_owned_receiver<R: Runtime>(client: Client) {
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    unsafe {
        kernel_copy_owned_receiver::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let actual = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&actual), [5, 8]);
}

#[cube(launch_unchecked)]
fn kernel_copy_receiver_before_args(output: &mut [u32]) {
    let mut cursor = Cursor {
        slot: 5u32.runtime(),
    };

    let snapshot = cursor.snapshot_and_bump({
        cursor.slot += 3u32;
        &mut cursor
    });

    output[0] = snapshot;
    output[1] = cursor.slot;
}

pub fn test_copy_receiver_before_args<R: Runtime>(client: Client) {
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    unsafe {
        kernel_copy_receiver_before_args::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let actual = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&actual), [5, 11]);
}

#[cube(launch_unchecked)]
fn kernel_copy_receiver_at_branch_start(input: &[u32], output: &mut [u32]) {
    let mut cursor = Cursor {
        slot: 5u32.runtime(),
    };

    if input[0] != 0u32 {
        let snapshot = cursor.snapshot_and_bump({
            cursor.slot += 3u32;
            &mut cursor
        });
        output[0] = snapshot;
    } else {
        let snapshot = cursor.snapshot_and_bump({
            cursor.slot += 6u32;
            &mut cursor
        });
        output[0] = snapshot;
    }

    output[1] = cursor.slot;
}

#[cube]
impl Cursor {
    fn plus(self, k: u32) -> u32 {
        self.slot + k
    }

    fn outer(self, k: u32) -> u32 {
        let mut c = Cursor {
            slot: self.slot + 1u32,
        };
        c.slot += k;

        // Path-call receiver created inside an outer method-syntax call.
        Cursor::plus(c, 10u32)
    }
}

#[cube(launch_unchecked)]
fn kernel_nested_path_receiver(input: &[u32], output: &mut [u32]) {
    let mut cursor = Cursor {
        slot: 5u32.runtime(),
    };
    cursor.slot += 1u32;
    output[0] = cursor.outer(input[0]);
    output[1] = cursor.slot;
}

pub fn test_nested_path_receiver<R: Runtime>(client: Client) {
    let input = client.create_from_slice(u32::as_bytes(&[2]));
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    unsafe {
        kernel_nested_path_receiver::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 1),
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let actual = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&actual), [19, 6]);
}

#[cube(launch_unchecked)]
fn kernel_closure_index_snapshot(input: &[u32], output: &mut [u32]) {
    let mut arr = Array::<u32>::new(2usize);
    arr[0] = input[0];

    let opt = ComptimeOption::<usize>::Some(0usize.runtime());
    let snapshot = opt.map_or::<u32, _>(0u32.runtime(), |i| arr[i]);

    arr[0] = 9u32;
    output[0] = snapshot;
    output[1] = arr[0];
}

pub fn test_closure_index_snapshot<R: Runtime>(client: Client) {
    let input = client.create_from_slice(u32::as_bytes(&[5]));
    let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

    unsafe {
        kernel_closure_index_snapshot::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 1),
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let actual = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&actual), [5, 9]);
}

pub fn test_copy_receiver_at_branch_start<R: Runtime>(client: Client) {
    for (flag, expected) in [(1u32, [5u32, 11]), (0u32, [5u32, 14])] {
        let input = client.create_from_slice(u32::as_bytes(&[flag]));
        let output = client.create_from_slice(u32::as_bytes(&[0, 0]));

        unsafe {
            kernel_copy_receiver_at_branch_start::launch_unchecked(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(input, 1),
                BufferArg::from_raw_parts(output.clone(), 2),
            );
        }

        let actual = client.read_one(output).unwrap();
        assert_eq!(u32::from_bytes(&actual), expected, "flag = {flag}");
    }
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
        mod assign {
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

            #[$crate::runtime_tests::test_log::test]
            fn test_copy_native_struct_field() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_copy_native_struct_field::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_copy_comptime_struct_field() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_copy_comptime_struct_field::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_mut_ref_struct_field() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_mut_ref_struct_field::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_copy_owned_receiver() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_copy_owned_receiver::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_copy_receiver_before_args() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_copy_receiver_before_args::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_copy_receiver_at_branch_start() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_copy_receiver_at_branch_start::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_nested_path_receiver() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_nested_path_receiver::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_closure_index_snapshot() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::assign::test_closure_index_snapshot::<TestRuntime>(client);
            }
        }
    };
}
