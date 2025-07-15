use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Ident, MatmulProblemSize};

#[derive(Clone, Debug)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct MatmulProblem {
    /// Number of rows in the output matrix
    pub m: usize,
    /// Number of columns in the output matrix
    pub n: usize,
    /// Reduction dimension
    pub k: usize,
    /// Batch shape for Lhs tensor
    pub lhs_batches: Vec<usize>,
    /// Batch shape for Rhs tensor
    pub rhs_batches: Vec<usize>,
    /// Memory layout of the Lhs matrix.
    pub lhs_layout: MatrixLayout,
    /// Memory layout of the Rhs matrix.
    pub rhs_layout: MatrixLayout,
}

impl MatmulProblem {
    /// Returns the batch dimensions of the output
    fn output_batch_dims(&self) -> Vec<usize> {
        self.lhs_batches
            .iter()
            .rev()
            .zip(self.rhs_batches.iter().rev())
            .map(|(&dim_lhs, &dim_rhs)| std::cmp::max(dim_lhs, dim_rhs))
            .collect()
    }

    /// Returns the total number of batches of the output
    pub(crate) fn num_batches(&self) -> usize {
        self.output_batch_dims().iter().product()
    }

    /// Returns the shape of the identified tensor, inferred by the problem definition
    #[allow(unused)]
    pub(crate) fn shape(&self, ident: Ident) -> Vec<usize> {
        match ident {
            Ident::Lhs => self
                .lhs_batches
                .iter()
                .cloned()
                .chain(vec![self.m, self.k])
                .collect(),
            Ident::Rhs => self
                .rhs_batches
                .iter()
                .cloned()
                .chain(vec![self.k, self.n])
                .collect(),
            Ident::Out => self
                .output_batch_dims()
                .iter()
                .cloned()
                .chain(vec![self.m, self.n])
                .collect(),
        }
    }
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Interpretation of matrix multiplication based on input shapes.
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

#[derive(CubeType, Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Layout of a 2D structure such as a tensor, shared memory or slice,
/// used within any matmul kernel level
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}

#[cube]
/// Maps the matmul MatrixLayout to cmma's MatrixLayout, for use in Cmma API.
pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
    match layout {
        MatrixLayout::RowMajor => cmma::MatrixLayout::RowMajor,
        MatrixLayout::ColMajor => cmma::MatrixLayout::ColMajor,
    }
}
