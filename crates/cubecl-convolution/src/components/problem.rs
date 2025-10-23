use cubecl_matmul::components::{MatmulProblem, MatrixLayout};

#[derive(Clone, Debug)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct ConvolutionProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,

    pub kernel_size: Vec<u32>,
    pub stride: Vec<u32>,
    pub padding: Vec<i32>,
    pub dilation: Vec<u32>,

    pub batches: usize,
    pub channels: usize,
    pub shape: Vec<usize>,
    pub out_shape: Vec<usize>,

    pub dimensionality: Dimensionality,
}

impl ConvolutionProblem {
    pub fn as_matmul_problem(&self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            lhs_batches: vec![],
            rhs_batches: vec![],
            out_batches: vec![],
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
        }
    }
}

/// Spatial dimensionality of an operation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Dimensionality {
    Dim1,
    Dim2,
    Dim3,
}

impl Dimensionality {
    pub fn num_dims(&self) -> u32 {
        match self {
            Dimensionality::Dim1 => 1,
            Dimensionality::Dim2 => 2,
            Dimensionality::Dim3 => 3,
        }
    }
}
