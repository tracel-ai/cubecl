use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::{ReinterpretSlice, ReinterpretSliceMut};
use half::f16;

#[cube(launch_unchecked)]
fn kernel_read_global(input: &Array<Line<i8>>, output: &mut Array<f16>) {
    let line_size = input.line_size();
    let list = ReinterpretSlice::<i8, f16>::new(input.to_slice(), line_size);
    output[UNIT_POS as usize] = list.read(UNIT_POS as usize);
}

pub fn run_test_read_global<R: Runtime>(client: ComputeClient<R>, line_size: usize) {
    if !client.properties().features.dynamic_line_size {
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
            ArrayArg::from_raw_parts::<i8>(input, 4 / line_size, line_size),
            ArrayArg::from_raw_parts::<f16>(output.clone(), 2, 1),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[cube(launch_unchecked)]
fn kernel_write_global(output: &mut Array<Line<i8>>, input: &Array<f16>) {
    let line_size = output.line_size();
    let mut list = ReinterpretSliceMut::<i8, f16>::new(output.to_slice_mut(), line_size);
    list.write(UNIT_POS as usize, input[UNIT_POS as usize]);
}

pub fn run_test_write_global<R: Runtime>(client: ComputeClient<R>, line_size: usize) {
    if !client.properties().features.dynamic_line_size {
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
            ArrayArg::from_raw_parts::<i8>(output.clone(), 4 / line_size, line_size),
            ArrayArg::from_raw_parts::<f16>(input, 2, 1),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = i8::from_bytes(&actual);

    assert_eq!(actual, casted);
}

#[cube(launch_unchecked)]
fn kernel_read_shared_memory(output: &mut Array<f16>) {
    let mut mem = SharedMemory::<i8>::new_lined(1usize, 4usize);
    if UNIT_POS == 0 {
        let mut line = Line::empty(4usize);
        line[0] = 0_i8;
        line[1] = 60_i8;
        line[2] = 64_i8;
        line[3] = -56_i8;
        mem[0] = line;
    }
    sync_cube();
    let list = ReinterpretSlice::<i8, f16>::new(mem.to_slice(), 4usize);
    output[UNIT_POS as usize] = list.read(UNIT_POS as usize);
}

pub fn run_test_read_shared_memory<R: Runtime>(client: ComputeClient<R>) {
    if !client.properties().features.dynamic_line_size {
        return; // can't run test
    }

    let target = [f16::from_f32(1.0), f16::from_f32(-8.5)];

    let output = client.empty(4);

    unsafe {
        kernel_read_shared_memory::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts::<f16>(output.clone(), 2, 1),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[cube(launch_unchecked)]
fn kernel_write_shared_memory(output: &mut Array<Line<i8>>, input: &Array<f16>) {
    let mut mem = SharedMemory::<i8>::new_lined(1usize, 4usize);
    let mut list = ReinterpretSliceMut::<i8, f16>::new(mem.to_slice_mut(), 4usize);
    let unit_pos = UNIT_POS as usize;
    list.write(unit_pos, input[unit_pos]);
    output[2 * unit_pos] = mem[2 * unit_pos];
    output[2 * unit_pos + 1] = mem[2 * unit_pos + 1];
}

pub fn run_test_write_shared_memory<R: Runtime>(client: ComputeClient<R>) {
    if !client.properties().features.dynamic_line_size {
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
            ArrayArg::from_raw_parts::<i8>(output.clone(), 1, 4),
            ArrayArg::from_raw_parts::<f16>(input, 2, 1),
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
