use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::ReinterpretList;
use half::f16;

#[cube(launch)]
fn kernel(input: &Array<Line<i8>>, output: &mut Array<f16>) {
    let line_size = input.line_size();
    let list = ReinterpretList::<i8, f16, &Array<Line<i8>>>::new(input, line_size);
    output[UNIT_POS] = list.read(UNIT_POS);
}

pub fn run_test<R: Runtime>(client: ComputeClient<R::Server, R::Channel>, line_size: usize) {
    let target = [f16::from_f32(1.0), f16::from_f32(-8.5)];
    let casted: [i8; 4] = unsafe { std::mem::transmute(target) };

    let input = client.create(i8::as_bytes(&casted));
    let output = client.empty(4);

    kernel::launch::<R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new(2, 1, 1),
        unsafe { ArrayArg::from_raw_parts::<i8>(&input, 4 / line_size, line_size as u8) },
        unsafe { ArrayArg::from_raw_parts::<f16>(&output, 2, 1) },
    );

    let actual = client.read_one(output.binding());
    let actual = f16::from_bytes(&actual);

    assert_eq!(actual, target);
}

#[macro_export]
macro_rules! testgen_reinterpret_list {
    () => {
        mod reinterpret_list_f16 {
            use super::*;

            #[test]
            fn from_i8x1() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test::<TestRuntime>(client, 1);
            }

            #[test]
            fn from_i8x2() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test::<TestRuntime>(client, 2);
            }

            #[test]
            fn from_i8x4() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::reinterpret_list::run_test::<TestRuntime>(client, 4);
            }
        }
    };
}
