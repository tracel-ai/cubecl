use cubecl_core::tensor_line_size_parallel;
use serde::{Deserialize, Serialize};

use crate::matmul::kernels::MatmulInvalidProblem;

use super::{Ident, MatmulProblemSize, MatrixLayout, batch};

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

    /// Asserts that the problem can be solved with the given batch matmul configs
    ///
    /// # Panics:
    ///
    ///  - If dimensions of the problem are larger than allowed by the config
    ///  - If line sizes do not divide well the dimension in which they are aligned
    pub fn check_config<B: batch::BatchConfig>(
        &self,
        config: &B,
    ) -> Result<(), MatmulInvalidProblem> {
        if self.m > config.max_m() as usize {
            return Err(MatmulInvalidProblem::ExceededMSize {
                m: self.m as u32,
                max_m: config.max_m(),
            });
        }

        if self.n > config.max_n() as usize {
            return Err(MatmulInvalidProblem::ExceededNSize {
                n: self.n as u32,
                max_n: config.max_n(),
            });
        }

        if self.num_batches() > config.max_batches() as usize {
            return Err(MatmulInvalidProblem::ExceededBatchSize {
                b: self.num_batches() as u32,
                max_b: config.max_batches(),
            });
        }

        Ok(())
    }

    pub fn check_line_sizes(
        &self,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulInvalidProblem> {
        match self.lhs_layout {
            MatrixLayout::RowMajor => {
                if self.k % line_sizes.lhs as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeLhs {
                        size: self.k as u32,
                        line_size: line_sizes.lhs,
                    });
                }
            }
            MatrixLayout::ColMajor => {
                if self.m % line_sizes.lhs as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeLhs {
                        size: self.m as u32,
                        line_size: line_sizes.lhs,
                    });
                }
            }
        }

        match self.rhs_layout {
            MatrixLayout::RowMajor => {
                if self.n % line_sizes.rhs as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeRhs {
                        size: self.n as u32,
                        line_size: line_sizes.rhs,
                    });
                }
            }
            MatrixLayout::ColMajor => {
                if self.k % line_sizes.rhs as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeRhs {
                        size: self.k as u32,
                        line_size: line_sizes.rhs,
                    });
                }
            }
        }

        if self.n % line_sizes.out as usize != 0 {
            return Err(MatmulInvalidProblem::InvalidLineSizeOut {
                size: self.n as u32,
                line_size: line_sizes.out,
            });
        }

        Ok(())
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
            matmul_size.m.into(),
            matmul_size.n.into(),
            matmul_size.k.into(),
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
        MatmulProblemSize {
            m: problem.m as u32,
            n: problem.n as u32,
            k: problem.k as u32,
        }
    }
}

impl From<MatmulProblem> for MatmulKind {
    fn from(problem: MatmulProblem) -> Self {
        MatmulProblemSize {
            m: problem.m as u32,
            n: problem.n as u32,
            k: problem.k as u32,
        }
        .into()
    }
}

impl From<&MatmulProblem> for MatmulKind {
    fn from(problem: &MatmulProblem) -> Self {
        MatmulProblemSize {
            m: problem.m as u32,
            n: problem.n as u32,
            k: problem.k as u32,
        }
        .into()
    }
}

#[derive(Clone, Debug)]
pub struct MatmulLineSizes {
    pub lhs: u8,
    pub rhs: u8,
    pub out: u8,
}

impl MatmulLineSizes {
    pub fn new_maximized(
        problem: &MatmulProblem,
        in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
    ) -> MatmulLineSizes {
        MatmulLineSizes {
            lhs: Self::maximize_lhs(problem, in_available.clone()),
            rhs: Self::maximize_rhs(problem, in_available),
            out: Self::maximize_out(problem, out_available),
        }
    }

    pub fn maximize_lhs(
        problem: &MatmulProblem,
        in_available: impl Iterator<Item = u8> + Clone,
    ) -> u8 {
        tensor_line_size_parallel(
            in_available.clone(),
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

    pub fn maximize_rhs(
        problem: &MatmulProblem,
        in_available: impl Iterator<Item = u8> + Clone,
    ) -> u8 {
        tensor_line_size_parallel(
            in_available,
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

    pub fn maximize_out(
        problem: &MatmulProblem,
        out_available: impl Iterator<Item = u8> + Clone,
    ) -> u8 {
        tensor_line_size_parallel(out_available, &[problem.k, problem.n], &[problem.n, 1], 1)
    }
}
