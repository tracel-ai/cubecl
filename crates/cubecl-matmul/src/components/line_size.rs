use cubecl_core::tensor_line_size_parallel;

use crate::components::{MatmulProblem, MatrixLayout};
use std::fmt::Debug;

pub struct MatmulLineSizes {
    pub lhs: u8,
    pub rhs: u8,
    pub out: u8,
}

#[derive(Clone, Debug)]
pub struct AvailableLineSizes {
    pub lhs: Vec<u8>,
    pub rhs: Vec<u8>,
    pub out: Vec<u8>,
}

impl AvailableLineSizes {
    pub fn maximize_lhs(&self, problem: &MatmulProblem, ceiling: Option<u8>) -> u8 {
        tensor_line_size_parallel(
            self.lhs
                .clone()
                .into_iter()
                .filter(|x| ceiling.map_or(true, |c| x <= &c)),
            &[problem.m, problem.k],
            &match problem.lhs_layout {
                MatrixLayout::RowMajor => [problem.k, 1],
                MatrixLayout::ColMajor => [1, problem.m],
            },
            match problem.lhs_layout {
                MatrixLayout::RowMajor => 1,
                MatrixLayout::ColMajor => 0,
            },
        )
    }

    pub fn maximize_rhs(&self, problem: &MatmulProblem, ceiling: Option<u8>) -> u8 {
        tensor_line_size_parallel(
            self.rhs
                .clone()
                .into_iter()
                .filter(|x| ceiling.map_or(true, |c| x <= &c)),
            &[problem.k, problem.n],
            &match problem.rhs_layout {
                MatrixLayout::RowMajor => [problem.n, 1],
                MatrixLayout::ColMajor => [1, problem.k],
            },
            match problem.rhs_layout {
                MatrixLayout::RowMajor => 1,
                MatrixLayout::ColMajor => 0,
            },
        )
    }

    pub fn maximize_out(&self, problem: &MatmulProblem, ceiling: Option<u8>) -> u8 {
        tensor_line_size_parallel(
            self.out
                .clone()
                .into_iter()
                .filter(|x| ceiling.map_or(true, |c| x <= &c)),
            &[problem.k, problem.n],
            &[problem.n, 1],
            1,
        )
    }
}
