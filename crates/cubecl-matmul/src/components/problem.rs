use serde::{Deserialize, Serialize};

use super::{Ident, MatmulProblemSize, MatrixLayout};

#[derive(Clone, Debug)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct MatmulProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batches: (Vec<usize>, Vec<usize>),
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
}

impl MatmulProblem {
    pub(crate) fn batch_dims(&self) -> Vec<usize> {
        self.batches
            .0
            .iter()
            .rev()
            .zip(self.batches.1.iter().rev())
            .map(|(&dim_lhs, &dim_rhs)| std::cmp::max(dim_lhs, dim_rhs))
            .collect()
    }

    /// Returns the total number of batches
    pub(crate) fn num_batches(&self) -> usize {
        self.batch_dims().iter().product()
    }

    /// Returns the shape of the identified tensor, inferred by the problem definition
    #[allow(unused)]
    pub(crate) fn shape(&self, ident: Ident) -> Vec<usize> {
        match ident {
            Ident::Lhs => self
                .batches
                .0
                .iter()
                .cloned()
                .chain(vec![self.m, self.k])
                .collect(),
            Ident::Rhs => self
                .batches
                .1
                .iter()
                .cloned()
                .chain(vec![self.k, self.n])
                .collect(),
            Ident::Out => self
                .batch_dims()
                .iter()
                .cloned()
                .chain(vec![self.m, self.n])
                .collect(),
        }
    }
}

/// Interpretation of matrix multiplication based on input shapes.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum MatmulKind {
    /// (M, K) @ (K, N) → (M, N), with M, K, N > 1
    General,

    /// (M, K) @ (K, 1) → (M, 1)
    MatVec,

    /// (1, K) @ (K, N) → (1, N)
    VecMat,

    /// (1, 1) @ (1, N) → (1, N)
    ScalarVec,

    /// (M, 1) @ (1, 1) → (M, 1)
    VecScalar,

    /// (1, K) @ (K, 1) → (1, 1)
    InnerProduct,

    /// (M, 1) @ (1, N) → (M, N)
    OuterProduct,

    /// (1, 1) @ (1, 1) → (1, 1)
    ScalarProduct,
}

impl From<MatmulProblemSize> for MatmulKind {
    fn from(matmul_size: MatmulProblemSize) -> Self {
        enum DimKind {
            Scalar,
            Vector,
        }

        impl From<u32> for DimKind {
            fn from(x: u32) -> Self {
                match x {
                    1 => DimKind::Scalar,
                    _ => DimKind::Vector,
                }
            }
        }

        use DimKind::*;
        match (
            matmul_size.m().into(),
            matmul_size.n().into(),
            matmul_size.k().into(),
        ) {
            (Scalar, Scalar, Scalar) => MatmulKind::ScalarProduct,
            (Scalar, Scalar, Vector) => MatmulKind::InnerProduct,
            (Scalar, Vector, Scalar) => MatmulKind::ScalarVec,
            (Scalar, Vector, Vector) => MatmulKind::VecMat,
            (Vector, Scalar, Scalar) => MatmulKind::VecScalar,
            (Vector, Scalar, Vector) => MatmulKind::MatVec,
            (Vector, Vector, Scalar) => MatmulKind::OuterProduct,
            (Vector, Vector, Vector) => MatmulKind::General,
        }
    }
}

impl From<MatmulProblem> for MatmulProblemSize {
    fn from(problem: MatmulProblem) -> Self {
        MatmulProblemSize::new(problem.m as u32, problem.n as u32, problem.k as u32)
    }
}

impl From<MatmulProblem> for MatmulKind {
    fn from(problem: MatmulProblem) -> Self {
        MatmulProblemSize::new(problem.m as u32, problem.n as u32, problem.k as u32).into()
    }
}

impl From<&MatmulProblem> for MatmulKind {
    fn from(problem: &MatmulProblem) -> Self {
        MatmulProblemSize::new(problem.m as u32, problem.n as u32, problem.k as u32).into()
    }
}
