use std::fmt::Display;

use cubecl_core::{
    ir::{Elem, FloatKind},
    prelude::Float,
    CubeElement, Runtime,
};

use crate::matmul::{
    cmma::{self, config::CmmaConfig, is_available},
    tests::matmul_test_case::MatmulTestCase,
};

use super::super::test_utils::assert_equals_approx;

#[derive(Copy, Clone)]
pub enum MatmulTest {
    One16_16_16,
    One32_16_8,
    One8_16_32,
    SmallRound,
    MediumMultibatch,
    LargeRound,
    SmallCheckBound,
    SmallVec2,
    SmallNoVec,
    MLargerThanN,
    MSmallerThanN,
}

impl From<MatmulTest> for MatmulTestCase {
    fn from(val: MatmulTest) -> Self {
        match val {
            MatmulTest::One16_16_16 => MatmulTestCase {
                m: 16,
                k: 16,
                n: 16,
                batch: 1,
            },
            MatmulTest::One32_16_8 => MatmulTestCase {
                m: 32,
                k: 16,
                n: 8,
                batch: 1,
            },
            MatmulTest::One8_16_32 => MatmulTestCase {
                m: 8,
                k: 16,
                n: 32,
                batch: 1,
            },
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

pub(crate) fn test_cmma<R: Runtime, F: Float + CubeElement + Display>(
    case: MatmulTestCase,
    config: CmmaConfig,
    device: &R::Device,
) -> Result<(), String> {
    let client = R::client(device);
    if is_available::<R, F>(&client, &config).is_ok() {
        let lhs = case.random_lhs::<R, F>(&client);
        let rhs = case.random_rhs::<R, F>(&client);

        let expected = case.matmul_cpu(&lhs, &rhs, &client);

        let out = cmma::launch::<R, F>(&client, lhs, rhs, case.empty_out(&client), config);

        // Lower required precision with f16/flex32
        let epsilon = match F::as_elem() {
            Elem::Float(FloatKind::F16) | Elem::Float(FloatKind::Relaxed) => 0.05,
            _ => 0.01,
        };

        assert_equals_approx::<R, F>(&client, out.handle, &expected, epsilon)
    } else {
        // Cmma unavailable, nothing to do
        Ok(())
    }
}
