use super::matrix::MatrixLayout;

#[derive(Copy, Clone)]
pub struct MatmulProblem {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,
}

impl MatmulProblem {
    pub fn new(
        m: u32,
        n: u32,
        k: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u8,
        rhs_line_size: u8,
        out_line_size: u8,
    ) -> MatmulProblem {
        MatmulProblem {
            m,
            n,
            k,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
