use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_absolute_pos(output1: &mut Array<u32>) {
    if ABSOLUTE_POS >= output1.len() {
        terminate!();
    }

    output1[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
}

pub fn test_kernel_topology_absolute_pos<R: Runtime>(client: ComputeClient<R>) {
    let cube_count = (3, 5, 7);
    let cube_dim = (16, 16, 1);

    let length = cube_count.0 * cube_count.1 * cube_count.2 * cube_dim.0 * cube_dim.1 * cube_dim.2;
    let handle1 = client.empty(length as usize * core::mem::size_of::<u32>());

    unsafe {
        kernel_absolute_pos::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, cube_count.2),
            CubeDim {
                x: cube_dim.0,
                y: cube_dim.1,
                z: cube_dim.2,
            },
            ArrayArg::from_raw_parts::<u32>(&handle1, length as usize, 1),
        )
        .unwrap()
    };

    let actual = client.read_one(handle1);
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = (0..length).collect();

    assert_eq!(actual, &expect);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_topology {
    () => {
        use super::*;

        #[test]
        fn test_topology_scalar() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::topology::test_kernel_topology_absolute_pos::<TestRuntime>(
                client,
            );
        }
    };
}
