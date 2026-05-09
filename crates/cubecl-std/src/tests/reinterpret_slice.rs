use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::{ReinterpretSlice, ReinterpretSliceMut};
use half::f16;

#[cube(launch_unchecked)]
fn kernel_read_global<N: Size>(input: &[Vector<i8, N>], output: &mut [f16]) {
    let list = ReinterpretSlice::<_, f16>::new(input);
    output[UNIT_POS as usize] = list.read(UNIT_POS as usize);
}

pub fn run_test_read_global<R: Runtime>(client: ComputeClient<R>, vector_size: usize) {
    if !client.features().memory_reinterpret {
        return; // can't run test
    }

    let target = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { core::mem::transmute(target) };

    let input = client.create_from_slice(i8::as_bytes(&casted));
    let output = client.empty(4);
    unsafe {
        kernel_read_global::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            vector_size,
            BufferArg::from_raw_parts(input, 4 / vector_size),
            BufferArg::from_raw_parts(output.clone(), 2),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[cube(launch_unchecked)]
fn kernel_write_global<N: Size>(output: &mut [Vector<i8, N>], input: &[f16]) {
    let mut list = ReinterpretSliceMut::<_, f16>::new(output);
    list.write(UNIT_POS as usize, input[UNIT_POS as usize]);
}

pub fn run_test_write_global<R: Runtime>(client: ComputeClient<R>, vector_size: usize) {
    if !client.features().memory_reinterpret {
        return; // can't run test
    }
    let source = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { core::mem::transmute(source) };

    let output = client.empty(4);
    let input = client.create_from_slice(f16::as_bytes(&source));

    unsafe {
        kernel_write_global::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            vector_size,
            BufferArg::from_raw_parts(output.clone(), 4 / vector_size),
            BufferArg::from_raw_parts(input, 2),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = i8::from_bytes(&actual);

    assert_eq!(actual, casted);
}

#[cube(launch_unchecked)]
fn kernel_read_shared_memory(output: &mut [f16]) {
    let mut mem = SharedMemory::<Vector<i8, Const<4>>>::new(1usize);
    if UNIT_POS == 0 {
        let mut vector = Vector::empty();
        vector.insert(0, 0_i8);
        vector.insert(1, 60_i8);
        vector.insert(2, 64_i8);
        vector.insert(3, -56_i8);
        mem[0] = vector;
    }
    sync_cube();
    let list = ReinterpretSlice::<_, f16>::new(mem.as_slice());
    output[UNIT_POS as usize] = list.read(UNIT_POS as usize);
}

pub fn run_test_read_shared_memory<R: Runtime>(client: ComputeClient<R>) {
    if !client.features().memory_reinterpret {
        return; // can't run test
    }

    let target = [f16::from_f32(1.0), f16::from_f32(-8.5)];

    let output = client.empty(4);

    unsafe {
        kernel_read_shared_memory::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            BufferArg::from_raw_parts(output.clone(), 2),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[cube(launch_unchecked)]
fn kernel_write_shared_memory<N: Size>(output: &mut [Vector<i8, N>], input: &[f16]) {
    let mut mem = SharedMemory::<Vector<i8, N>>::new(1usize);
    let mut list = ReinterpretSliceMut::<_, f16>::new(mem.as_mut_slice());
    let unit_pos = UNIT_POS as usize;
    list.write(unit_pos, input[unit_pos]);
    output[2 * unit_pos] = mem[2 * unit_pos];
    output[2 * unit_pos + 1] = mem[2 * unit_pos + 1];
}

pub fn run_test_write_shared_memory<R: Runtime>(client: ComputeClient<R>) {
    if !client.features().memory_reinterpret {
        return; // can't run test
    }

    let source = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { core::mem::transmute(source) };

    let output = client.empty(4);
    let input = client.create_from_slice(f16::as_bytes(&source));

    unsafe {
        kernel_write_shared_memory::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            4,
            BufferArg::from_raw_parts(output.clone(), 1),
            BufferArg::from_raw_parts(input, 2),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = i8::from_bytes(&actual);

    assert_eq!(actual, casted);
}

#[macro_export]
macro_rules! testgen_reinterpret_slice {
    () => {
        mod reinterpret_slice_f16 {
            use super::*;


            mod global {
                use super::*;

                #[$crate::tests::test_log::test]
                fn read_from_i8x1() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(client, 1);
                }

                #[$crate::tests::test_log::test]
                fn read_from_i8x2() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(client, 2);
                }

                #[$crate::tests::test_log::test]
                fn read_from_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(client, 4);
                }

                #[$crate::tests::test_log::test]
                fn write_into_i8x1() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(client, 1);
                }

                #[$crate::tests::test_log::test]
                fn write_into_i8x2() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(client, 2);
                }

                #[$crate::tests::test_log::test]
                fn write_into_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(client, 4);
                }
            }

            mod shared_memory {
                use super::*;

                #[$crate::tests::test_log::test]
                fn read_from_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_shared_memory::<TestRuntime>(client);
                }

                #[$crate::tests::test_log::test]
                fn write_from_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_shared_memory::<TestRuntime>(client);
                }
            }
        }
    };
}
