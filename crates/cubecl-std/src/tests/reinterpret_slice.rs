use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::{ReinterpretSlice, ReinterpretSliceMut};
use half::f16;

#[cube(launch_unchecked)]
fn kernel_read_global(input: &Array<Line<i8>>, output: &mut Array<f16>) {
    let line_size = input.line_size();
    let list = ReinterpretSlice::<i8, f16>::new(input.to_slice(), line_size);
    output[UNIT_POS] = list.read(UNIT_POS);
}

pub fn run_test_read_global<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    line_size: usize,
) {
    if !client
        .properties()
        .feature_enabled(cubecl_core::Feature::DynamicLineSize)
    {
        return; // can't run test
    }

    let target = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { core::mem::transmute(target) };

    let input = client.create(i8::as_bytes(&casted));
    let output = client.empty(4);
    unsafe {
        kernel_read_global::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts::<i8>(&input, 4 / line_size, line_size as u8),
            ArrayArg::from_raw_parts::<f16>(&output, 2, 1),
        );
    }

    let actual = client.read_one(output.binding());
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[cube(launch_unchecked)]
fn kernel_write_global(output: &mut Array<Line<i8>>, input: &Array<f16>) {
    let line_size = output.line_size();
    let mut list = ReinterpretSliceMut::<i8, f16>::new(output.to_slice_mut(), line_size);
    list.write(UNIT_POS, input[UNIT_POS]);
}

pub fn run_test_write_global<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    line_size: usize,
) {
    if !client
        .properties()
        .feature_enabled(cubecl_core::Feature::DynamicLineSize)
    {
        return; // can't run test
    }
    let source = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { core::mem::transmute(source) };

    let output = client.empty(4);
    let input = client.create(f16::as_bytes(&source));

    unsafe {
        kernel_write_global::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts::<i8>(&output, 4 / line_size, line_size as u8),
            ArrayArg::from_raw_parts::<f16>(&input, 2, 1),
        );
    }

    let actual = client.read_one(output.binding());
    let actual = i8::from_bytes(&actual);

    assert_eq!(actual, casted);
}

#[cube(launch_unchecked)]
fn kernel_read_shared_memory(output: &mut Array<f16>) {
    let mut mem = SharedMemory::<i8>::new_lined(1_u32, 4_u32);
    if UNIT_POS == 0 {
        let mut line = Line::empty(4_u32);
        line[0] = 0_i8;
        line[1] = 60_i8;
        line[2] = 64_i8;
        line[3] = -56_i8;
        mem[0] = line;
    }
    sync_cube();
    let list = ReinterpretSlice::<i8, f16>::new(mem.to_slice(), 4_u32);
    output[UNIT_POS] = list.read(UNIT_POS);
}

pub fn run_test_read_shared_memory<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    if !client
        .properties()
        .feature_enabled(cubecl_core::Feature::DynamicLineSize)
    {
        return; // can't run test
    }

    let target = [f16::from_f32(1.0), f16::from_f32(-8.5)];

    let output = client.empty(4);

    unsafe {
        kernel_read_shared_memory::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts::<f16>(&output, 2, 1),
        );
    }

    let actual = client.read_one(output.binding());
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[cube(launch_unchecked)]
fn kernel_write_shared_memory(output: &mut Array<Line<i8>>, input: &Array<f16>) {
    let mut mem = SharedMemory::<i8>::new_lined(1_u32, 4_u32);
    let mut list = ReinterpretSliceMut::<i8, f16>::new(mem.to_slice_mut(), 4_u32);
    list.write(UNIT_POS, input[UNIT_POS]);
    output[2 * UNIT_POS] = mem[2 * UNIT_POS];
    output[2 * UNIT_POS + 1] = mem[2 * UNIT_POS + 1];
}

pub fn run_test_write_shared_memory<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    if !client
        .properties()
        .feature_enabled(cubecl_core::Feature::DynamicLineSize)
    {
        return; // can't run test
    }

    let source = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { core::mem::transmute(source) };

    let output = client.empty(4);
    let input = client.create(f16::as_bytes(&source));

    unsafe {
        kernel_write_shared_memory::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts::<i8>(&output, 1, 4),
            ArrayArg::from_raw_parts::<f16>(&input, 2, 1),
        );
    }

    let actual = client.read_one(output.binding());
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

                #[test]
                fn read_from_i8x1() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(client, 1);
                }

                #[test]
                fn read_from_i8x2() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(client, 2);
                }

                #[test]
                fn read_from_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(client, 4);
                }

                #[test]
                fn write_into_i8x1() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(client, 1);
                }

                #[test]
                fn write_into_i8x2() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(client, 2);
                }

                #[test]
                fn write_into_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(client, 4);
                }
            }

            mod shared_memory {
                use super::*;

                #[test]
                fn read_from_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_read_shared_memory::<TestRuntime>(client);
                }

                #[test]
                fn write_from_i8x4() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::reinterpret_slice::run_test_write_shared_memory::<TestRuntime>(client);
                }
            }
        }
    };
}
