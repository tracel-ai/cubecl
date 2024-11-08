use std::marker::PhantomData;

use cubecl_core::prelude::Numeric;

use super::{batch, MatrixLayout};

#[derive(Clone)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct MatmulProblem<EG: Numeric> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batches: Vec<usize>,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,
    pub _element: PhantomData<EG>,
}

impl<EG: Numeric> MatmulProblem<EG> {
    /// Returns the total number of batches
    pub(crate) fn num_batches(&self) -> usize {
        self.batches.iter().copied().product()
    }

    /// Asserts that the problem can be solved with the given batch matmul configs
    ///
    /// # Panics:
    ///
    ///  - If dimensions of the problem are larger than allowed by the config
    ///  - If line sizes do not divide well the dimension in which they are aligned
    pub(crate) fn check_config<B: batch::Config>(&self, config: &B) {
        assert!(
            self.m <= config.max_m() as usize,
            "Problem has m={} but these configs can only have m<={}",
            self.m,
            config.max_m()
        );
        assert!(
            self.n <= config.max_n() as usize,
            "Problem has n={} but these configs can only have n<={}",
            self.n,
            config.max_n()
        );
        assert!(
            self.num_batches() <= config.max_batches() as usize,
            "Problem has {} batches but these configs can only have batches<={}",
            self.num_batches(),
            config.max_batches()
        );

        assert!(match self.lhs_layout {
            MatrixLayout::RowMajor => self.k % self.lhs_line_size as usize == 0,
            MatrixLayout::ColMajor => self.m % self.lhs_line_size as usize == 0,
        });

        assert!(match self.rhs_layout {
            MatrixLayout::RowMajor => self.n % self.rhs_line_size as usize == 0,
            MatrixLayout::ColMajor => self.k % self.rhs_line_size as usize == 0,
        });

        assert!(self.n % self.out_line_size as usize == 0);
    }
}
