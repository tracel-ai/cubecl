use cubecl_core::Runtime;

use crate::matmul::{
    cmma::{config::CmmaConfig, launch},
    tests::matmul_test_case::MatmulTestCase,
};

use super::super::test_utils::{assert_equals_approx, cmma_available};

pub fn test_matmul_cmma_one_cube<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
    };

    test_cmma::<R>(&case, CmmaConfig::default(), device);
}

macro_rules! alternate_block_sizes {
    ($name:ident, $b_mn:expr, $b_k:expr) => {
        pub fn $name<R: Runtime>(device: &R::Device) {
            let case = MatmulTestCase {
                m: 128,
                k: 128,
                n: 128,
                batch: 1,
            };
            test_cmma::<R>(
                &case,
                CmmaConfig {
                    b_mn: $b_mn,
                    b_k: $b_k,
                    ..Default::default()
                },
                device,
            );
        }
    };
}

alternate_block_sizes!(test_matmul_cmma_16_16, 16, 16);
alternate_block_sizes!(test_matmul_cmma_32_16, 32, 16);
alternate_block_sizes!(test_matmul_cmma_32_32, 32, 32);
alternate_block_sizes!(test_matmul_cmma_64_16, 64, 16);
alternate_block_sizes!(test_matmul_cmma_64_32, 64, 32);
alternate_block_sizes!(test_matmul_cmma_64_64, 64, 64);
alternate_block_sizes!(test_matmul_cmma_128_16, 128, 16);
alternate_block_sizes!(test_matmul_cmma_128_32, 128, 32);

pub fn test_matmul_cmma_several_cubes<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
    };

    test_cmma::<R>(&case, CmmaConfig::default(), device);
}

pub fn test_matmul_cmma_with_check_bounds<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
    };

    test_cmma::<R>(&case, CmmaConfig::default(), device);
}

pub fn test_matmul_cmma_with_batches<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
    };

    test_cmma::<R>(&case, CmmaConfig::default(), device);
}

pub fn test_matmul_cmma_unvectorizable_shapes<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 63,
        k: 63,
        n: 63,
        batch: 3,
    };

    test_cmma::<R>(&case, CmmaConfig::default(), device);
}

pub fn test_matmul_cmma_vec2_shapes<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 62,
        k: 62,
        n: 62,
        batch: 3,
    };

    test_cmma::<R>(&case, CmmaConfig::default(), device);
}

fn test_cmma<R: Runtime>(case: &MatmulTestCase, config: CmmaConfig, device: &R::Device) {
    if !cmma_available::<R>(device) {
        // We can't execute the test, skip.
        return;
    }

    let client = R::client(device);
    let lhs = case.random_lhs::<R>(&client);
    let rhs = case.random_rhs::<R>(&client);

    let expected = case.matmul_cpu(&lhs, &rhs, &client);

    let out = launch::<R, f32>(&client, lhs, rhs, case.empty_out(&client), config);

    assert_equals_approx::<R>(&client, out.handle, &expected, 10e-3);
}
