use cubecl_core::Runtime;

use crate::matmul::{
    cmma::{
        config::{CmmaConfig, PredefinedCmmaConfig},
        launch,
    },
    tests::matmul_test_case::MatmulTestCase,
};

use super::super::test_utils::{assert_equals_approx, cmma_available};

pub enum MatmulTest {
    SmallRound,
    MediumMultibatch,
    LargeRound,
    SmallCheckBound,
    SmallVec2,
    SmallNoVec,
    MLargerThanN,
    MSmallerThanN,
}

impl Into<MatmulTestCase> for MatmulTest {
    fn into(self) -> MatmulTestCase {
        match self {
            MatmulTest::SmallRound => MatmulTestCase {
                m: 64,
                k: 64,
                n: 64,
                batch: 1,
            },
            MatmulTest::MediumMultibatch => MatmulTestCase {
                m: 128,
                k: 64,
                n: 128,
                batch: 3,
            },
            MatmulTest::LargeRound => MatmulTestCase {
                m: 256,
                k: 256,
                n: 256,
                batch: 1,
            },
            MatmulTest::SmallCheckBound => MatmulTestCase {
                m: 60,
                k: 60,
                n: 60,
                batch: 1,
            },
            MatmulTest::SmallVec2 => MatmulTestCase {
                m: 62,
                k: 62,
                n: 62,
                batch: 1,
            },
            MatmulTest::SmallNoVec => MatmulTestCase {
                m: 63,
                k: 63,
                n: 63,
                batch: 1,
            },
            MatmulTest::MLargerThanN => MatmulTestCase {
                m: 256,
                k: 64,
                n: 64,
                batch: 1,
            },
            MatmulTest::MSmallerThanN => MatmulTestCase {
                m: 64,
                k: 64,
                n: 256,
                batch: 1,
            },
        }
    }
}

pub fn test_temp<R: Runtime>(device: &R::Device) {
    test_cmma::<R>(
        MatmulTest::SmallRound.into(),
        PredefinedCmmaConfig::M128K16.into(),
        device,
    );
}

fn test_cmma<R: Runtime>(case: MatmulTestCase, config: CmmaConfig, device: &R::Device) {
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
