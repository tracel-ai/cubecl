use crate::matmul::kernels::MatmulInvalidProblem;

use super::{MatrixLayout, batch};

#[derive(Clone, Debug)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct MatmulProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batches: (Vec<usize>, Vec<usize>),
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,
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

        match self.lhs_layout {
            MatrixLayout::RowMajor => {
                if self.k % self.lhs_line_size as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeLhs {
                        size: self.k as u32,
                        line_size: self.lhs_line_size,
                    });
                }
            }
            MatrixLayout::ColMajor => {
                if self.m % self.lhs_line_size as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeLhs {
                        size: self.m as u32,
                        line_size: self.lhs_line_size,
                    });
                }
            }
        }

        match self.rhs_layout {
            MatrixLayout::RowMajor => {
                if self.n % self.rhs_line_size as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeRhs {
                        size: self.n as u32,
                        line_size: self.rhs_line_size,
                    });
                }
            }
            MatrixLayout::ColMajor => {
                if self.k % self.rhs_line_size as usize != 0 {
                    return Err(MatmulInvalidProblem::InvalidLineSizeRhs {
                        size: self.k as u32,
                        line_size: self.lhs_line_size,
                    });
                }
            }
        }

        if self.n % self.out_line_size as usize != 0 {
            return Err(MatmulInvalidProblem::InvalidLineSizeOut {
                size: self.n as u32,
                line_size: self.out_line_size,
            });
        }

        Ok(())
    }
}
