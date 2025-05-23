use crate as cubecl;
use crate::prelude::*;

#[cube(launch)]
fn unit_pos_write<F: Float>(out: &mut Array<F>) {
    out[UNIT_POS] = F::cast_from(UNIT_POS);
}

pub fn test_unit_pos<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let x = 32;
    let y = 32;
    let input_size = x * y;
    let handle = client.empty(core::mem::size_of::<f32>() * input_size);

    unit_pos_write::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(x as u32, y as u32),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, input_size, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    println!("{:?}", actual);
    assert!(false);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_unit_pos {
    () => {
        use super::*;

        #[test]
        fn test_unit_pos() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::unit_pos::test_unit_pos::<TestRuntime>(client);
        }
    };
}
