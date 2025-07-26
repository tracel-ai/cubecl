use crate::{self as cubecl};
use crate::{Feature, prelude::*};

#[cube(launch, cluster_dim = CubeDim::new_3d(1, 2, 3))]
fn cluster_meta_kernel(out: &mut Array<u32>) {
    if UNIT_POS == 0 {
        if CUBE_POS == 0 {
            out[0] = CUBE_CLUSTER_DIM;
            out[1] = CUBE_CLUSTER_DIM_X;
            out[2] = CUBE_CLUSTER_DIM_Y;
            out[3] = CUBE_CLUSTER_DIM_Z;
        }

        let offset = CUBE_POS * 4 + 4;

        out[offset] = CUBE_POS_CLUSTER;
        out[offset + 1] = CUBE_POS_CLUSTER_X;
        out[offset + 2] = CUBE_POS_CLUSTER_Y;
        out[offset + 3] = CUBE_POS_CLUSTER_Z;
    }
}

pub fn test_cluster_meta<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    if !client.properties().feature_enabled(Feature::CubeCluster) {
        return;
    }

    let cluster_dim_x = 1;
    let cluster_dim_y = 2;
    let cluster_dim_z = 3;

    let cube_count_x = 2;
    let cube_count_y = 2;
    let cube_count_z = 6;
    let cube_count = CubeCount::new_3d(cube_count_x, cube_count_y, cube_count_z);
    let num_cubes = cube_count_x * cube_count_y * cube_count_z;

    let handle = client
        .empty((num_cubes as usize * 4 + 4) * size_of::<u32>())
        .expect("Alloc failed");

    let vectorization = 1;

    cluster_meta_kernel::launch::<R>(&client, cube_count, CubeDim::new_single(), unsafe {
        ArrayArg::from_raw_parts::<f32>(&handle, num_cubes as usize * 8, vectorization)
    });

    let actual = client.read_one(handle.binding());
    let actual = u32::from_bytes(&actual);

    let mut expected: Vec<u32> = vec![6, 1, 2, 3];
    for z in 0..cube_count_z {
        for y in 0..cube_count_y {
            for x in 0..cube_count_x {
                let rank_x = x % cluster_dim_x;
                let rank_y = y % cluster_dim_y;
                let rank_z = z % cluster_dim_z;
                let rank_abs = rank_z * cluster_dim_y + rank_y * cluster_dim_x + rank_x;
                expected.extend([rank_abs, rank_x, rank_y, rank_z]);
            }
        }
    }

    assert_eq!(actual, &expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cluster {
    () => {
        use super::*;

        #[test]
        fn test_cluster_meta() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::cluster::test_cluster_meta::<TestRuntime>(client);
        }
    };
}
