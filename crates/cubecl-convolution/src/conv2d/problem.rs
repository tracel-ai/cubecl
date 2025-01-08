use cubecl_linalg::matmul::components::{MatmulProblem, MatrixLayout};

use crate::ConvOptions;

#[derive(Clone)]
/// Description of a convolution problem to solve, regardless of actual data
pub struct ConvolutionProblem {
    // Underlying matmul constraints
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,

    // Convolution constraints
    pub kernel_size: (u32, u32),
    pub options: ConvOptions<2>,
    pub out_shape_y: usize,
    pub out_shape_x: usize,
    pub has_bias: bool,
}

impl ConvolutionProblem {
    pub fn as_matmul_problem(&self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            batches: (vec![], vec![]),
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
            lhs_line_size: self.lhs_line_size,
            rhs_line_size: self.rhs_line_size,
            out_line_size: self.out_line_size,
        }
    }
}
