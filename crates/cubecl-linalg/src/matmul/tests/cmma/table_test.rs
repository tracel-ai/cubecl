use cubecl_core::Runtime;
use std::io::{self, Write};

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
            0 => Some((PredefinedCmmaConfig::M128K16, "m128_k16".to_string())),
            1 => Some((PredefinedCmmaConfig::M64K32, "m64_k32".to_string())),
            2 => Some((PredefinedCmmaConfig::M64K16, "m64_k16".to_string())),
            3 => Some((PredefinedCmmaConfig::M32K16, "m32_k16".to_string())),
            4 => Some((PredefinedCmmaConfig::M32K32, "m32_k32".to_string())),
            5 => Some((
                PredefinedCmmaConfig::SplitM32k32,
                "split_m32_k32".to_string(),
            )),
            6 => Some((
                PredefinedCmmaConfig::SplitM64k16,
                "split_m64_k16".to_string(),
            )),
            7 => Some((
                PredefinedCmmaConfig::TilewiseInverted,
                "tilewise_inverted".to_string(),
            )),
            8 => Some((PredefinedCmmaConfig::Continuous, "continuous".to_string())),
            9 => Some((
                PredefinedCmmaConfig::ContinuousInverted,
                "continuous_inverted".to_string(),
            )),
            10 => Some((PredefinedCmmaConfig::LargeSmem, "large_smem".to_string())),
            11 => Some((
                PredefinedCmmaConfig::RowMajorRasterization,
                "row_major_dispatch".to_string(),
            )),
            12 => Some((
                PredefinedCmmaConfig::SwizzleRasterization,
                "swizzle_dispatch".to_string(),
            )),
            13 => Some((
                PredefinedCmmaConfig::AccumulatorsFirstNoReuse,
                "accumulators_first_no_reuse".to_string(),
            )),
            14 => Some((
                PredefinedCmmaConfig::BuffersFirst,
                "buffers_first".to_string(),
            )),
            16 => Some((PredefinedCmmaConfig::M16K32N64, "m_16_k32_n64".to_string())),
            17 => Some((PredefinedCmmaConfig::M32K16N64, "m_32_k16_n64".to_string())),
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
            8 => Some((MatmulTest::One16_16_16, "one_16_16_16".to_string())),
            9 => Some((MatmulTest::One32_16_8, "one_32_16_8".to_string())),
            10 => Some((MatmulTest::One8_16_32, "one_8_16_32".to_string())),
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
            let test_name = format!("test_{}_{}", case_name, config_name);
            combinations.push((case, config, test_name));
        }
    }

    combinations
}

/// This is a special test that encapsulates many sub tests
/// To filter among them you can specify the TEST_FILTER environment variable
/// It should be run with --nocapture
pub fn test_cmma_all<R: Runtime>(device: &R::Device) {
    let filter = std::env::var("TEST_FILTER").unwrap_or_default();
    let mut all_ok = true;
    let mut n_tests_run = 0;

    for (test_case, config, name) in all_combinations() {
        if filter.is_empty() || name.contains(&filter) {
            print!("Running test {}...", name);
            io::stdout().flush().unwrap();

            match test_cmma::<R>(test_case.into(), config.into(), device) {
                Ok(_) => println!("Ok"),
                Err(e) => {
                    all_ok = false;
                    println!("Failure: {}", e)
                }
            }
            io::stdout().flush().unwrap();

            n_tests_run += 1;
        }
    }

    assert!(
        n_tests_run > 0,
        "No sub tests run, is env variable TEST_FILTER a valid filter?"
    );
    assert!(all_ok, "Some tests failed");
}
