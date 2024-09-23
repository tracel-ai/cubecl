use cubecl_core::Runtime;

use crate::matmul::cmma::config::PredefinedCmmaConfig;

use super::test_cases::{test_cmma, MatmulTest};

struct CmmaConfigIterator {
    current: usize,
}

struct TestCaseIterator {
    current: usize,
}

impl CmmaConfigIterator {
    fn new() -> Self {
        CmmaConfigIterator { current: 0 }
    }
}

impl TestCaseIterator {
    fn new() -> Self {
        TestCaseIterator { current: 0 }
    }
}

impl Iterator for CmmaConfigIterator {
    type Item = (PredefinedCmmaConfig, String);

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.current {
            0 => Some((PredefinedCmmaConfig::M128K32, "m128_k32".to_string())),
            1 => Some((PredefinedCmmaConfig::M128K16, "m128_k16".to_string())),
            2 => Some((PredefinedCmmaConfig::M64K32, "m64_k32".to_string())),
            3 => Some((PredefinedCmmaConfig::M64K16, "m64_k16".to_string())),
            4 => Some((PredefinedCmmaConfig::M32K16, "m32_k16".to_string())),
            5 => Some((PredefinedCmmaConfig::M32K32, "m32_k32".to_string())),
            6 => Some((PredefinedCmmaConfig::SplitM32k32, "m32_k32".to_string())),
            7 => Some((PredefinedCmmaConfig::SplitM128k16, "m128_k16".to_string())),
            8 => Some((
                PredefinedCmmaConfig::TilewiseInverted,
                "tilewise_inverted".to_string(),
            )),
            9 => Some((PredefinedCmmaConfig::Continuous, "continuous".to_string())),
            10 => Some((
                PredefinedCmmaConfig::ContinuousInverted,
                "continuous_inverted".to_string(),
            )),
            11 => Some((PredefinedCmmaConfig::LargeSmem, "large_smem".to_string())),
            12 => Some((
                PredefinedCmmaConfig::RowMajorDispatch,
                "row_major_dispatch".to_string(),
            )),
            13 => Some((
                PredefinedCmmaConfig::SwizzleDispatch,
                "swizzle_dispatch".to_string(),
            )),
            14 => Some((
                PredefinedCmmaConfig::AccumulatorsFirstNoReuse,
                "accumulators_no_reuse".to_string(),
            )),
            15 => Some((
                PredefinedCmmaConfig::AccumulatorsFirstWithReuse,
                "accumulators_reuse".to_string(),
            )),
            _ => None,
        };

        self.current += 1;
        result
    }
}

impl Iterator for TestCaseIterator {
    type Item = (MatmulTest, String);

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.current {
            0 => Some((MatmulTest::SmallRound, "small_round".to_string())),
            1 => Some((
                MatmulTest::MediumMultibatch,
                "medium_multibatch".to_string(),
            )),
            2 => Some((MatmulTest::LargeRound, "large_round".to_string())),
            3 => Some((MatmulTest::SmallCheckBound, "small_check_bound".to_string())),
            4 => Some((MatmulTest::SmallVec2, "small_vec2".to_string())),
            5 => Some((MatmulTest::SmallNoVec, "small_novec".to_string())),
            6 => Some((MatmulTest::MLargerThanN, "m_larger_than_n".to_string())),
            7 => Some((MatmulTest::MSmallerThanN, "m_smaller_than_n".to_string())),
            _ => None,
        };

        self.current += 1;
        result
    }
}

fn all_combinations() -> Vec<(MatmulTest, PredefinedCmmaConfig, String)> {
    let mut combinations = Vec::new();
    let test_cases = TestCaseIterator::new();

    for (case, case_name) in test_cases {
        let test_configs = CmmaConfigIterator::new();
        for (config, config_name) in test_configs {
            let test_name = format!("test_{}_with_{}", case_name, config_name);
            combinations.push((case, config, test_name));
        }
    }

    combinations
}

pub fn test_cmma_all<R: Runtime>(device: &R::Device) {
    let mut all_ok = true;

    for (test_case, config, name) in all_combinations() {
        println!("Running test {}", name);

        match test_cmma::<R>(test_case.into(), config.into(), device) {
            Ok(_) => println!("Ok"),
            Err(e) => {
                all_ok = false;
                println!("Failure: {}", e)
            }
        }

        assert!(all_ok)
    }
}
