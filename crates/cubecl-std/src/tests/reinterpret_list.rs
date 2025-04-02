use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::ReinterpretList;
use half::f16;

#[cube(launch)]
fn kernel_read(input: &Array<Line<i8>>, output: &mut Array<f16>) {
    let line_size = input.line_size();
    let list = ReinterpretList::<i8, f16, &Array<Line<i8>>>::new(input, line_size);
    output[UNIT_POS] = list.read(UNIT_POS);
}

pub fn run_test_read<R: Runtime>(client: ComputeClient<R::Server, R::Channel>, line_size: usize) {
    let target = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { std::mem::transmute(target) };

    let input = client.create(i8::as_bytes(&casted));
    let output = client.empty(4);

    kernel_read::launch::<R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(2),
        unsafe { ArrayArg::from_raw_parts::<i8>(&input, 4 / line_size, line_size as u8) },
        unsafe { ArrayArg::from_raw_parts::<f16>(&output, 2, 1) },
    );

    let actual = client.read_one(output.binding());
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[cube(launch)]
fn kernel_write(output: &mut Array<Line<i8>>, input: &Array<f16>) {
    let line_size = output.line_size();
    let list = ReinterpretList::<i8, f16, &Array<Line<i8>>>::new(output, line_size);
    list.write(UNIT_POS, input[UNIT_POS]);
}

pub fn run_test_write<R: Runtime>(client: ComputeClient<R::Server, R::Channel>, line_size: usize) {
    let source = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { std::mem::transmute(source) };

    let output = client.empty(4);
    let input = client.create(f16::as_bytes(&source));

    kernel_read::launch::<R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(2),
        unsafe { ArrayArg::from_raw_parts::<i8>(&output, 4 / line_size, line_size as u8) },
        unsafe { ArrayArg::from_raw_parts::<f16>(&input, 2, 1) },
    );

    let actual = client.read_one(output.binding());
    let actual = i8::from_bytes(&actual);

    assert_eq!(actual, casted);
}

#[macro_export]
macro_rules! testgen_reinterpret_list {
    () => {
        mod reinterpret_list_f16 {
            use super::*;

            #[test]
            fn read_from_i8x1() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test_read::<TestRuntime>(client, 1);
            }

            #[test]
            fn read_from_i8x2() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test_read::<TestRuntime>(client, 2);
            }

            #[test]
            fn read_from_i8x4() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test_read::<TestRuntime>(client, 4);
            }

            #[test]
            fn write_into_i8x1() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test_write::<TestRuntime>(client, 1);
            }

            #[test]
            fn write_into_i8x2() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test_write::<TestRuntime>(client, 2);
            }

            #[test]
            fn write_into_i8x4() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test_write::<TestRuntime>(client, 4);
            }
        }
    };
}
