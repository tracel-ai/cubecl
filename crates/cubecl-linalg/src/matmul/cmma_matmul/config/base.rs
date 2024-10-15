use cubecl_core::prelude::*;

use crate::matmul::{config::Config, problem::MatmulProblem};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CmmaConfig {
    pub out_smem_line_size: u32,
}

impl Config for CmmaConfig {
    type ProblemDefinition = MatmulProblem;

    fn from_problem(problem: Self::ProblemDefinition) -> Self {
        Self {
            out_smem_line_size: problem.out_line_size as u32,
        }
    }
}

impl Init for CmmaConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl CubeType for CmmaConfig {
    type ExpandType = Self;
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self {
            out_smem_line_size: 4u32,
        }
    }
}
