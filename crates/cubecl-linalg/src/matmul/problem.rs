use super::matrix_layout::MatrixLayout;

#[derive(Copy, Clone)]
pub struct MatmulProblem {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
}

impl MatmulProblem {
    pub fn new(
        m: u32,
        n: u32,
        k: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
    ) -> MatmulProblem {
        MatmulProblem {
            m,
            n,
            k,
            lhs_layout,
            rhs_layout,
        }
    }
}

pub struct Requirements {
    pub num_planes: u32,
    pub num_cubes: u32,
}
