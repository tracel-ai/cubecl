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
