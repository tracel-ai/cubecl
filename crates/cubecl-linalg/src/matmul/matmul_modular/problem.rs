use std::marker::PhantomData;

use cubecl_core::prelude::Numeric;

#[cfg(feature = "export_tests")]
use super::matrix::Ident;
use super::{matmul_batch::BmmConfig, matrix::MatrixLayout};

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
    #[cfg(feature = "export_tests")]
    /// Returns the total number of elements for the identified tensor
    pub(crate) fn tensor_size(&self, ident: Ident) -> usize {
        match ident {
            Ident::Lhs => self.num_batches() * self.m * self.k,
            Ident::Rhs => self.num_batches() * self.k * self.n,
            Ident::Out => self.num_batches() * self.m * self.n,
        }
    }

    #[cfg(feature = "export_tests")]
    /// Returns the shape of the identified tensor
    pub(crate) fn shape(&self, ident: Ident) -> Vec<usize> {
        self.batches
            .iter()
            .cloned()
            .chain(
                match ident {
                    Ident::Lhs => vec![self.m, self.k],
                    Ident::Rhs => vec![self.k, self.n],
                    Ident::Out => vec![self.m, self.n],
                }
                .into_iter(),
            )
            .collect()
    }

    #[cfg(feature = "export_tests")]
    /// Returns the stride of the identified tensor
    pub(crate) fn strides(&self, ident: Ident) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.batches.len() + 2);

        let (last_batch, x, y) = match ident {
            Ident::Lhs => match self.lhs_layout {
                MatrixLayout::RowMajor => (self.m * self.k, self.k, 1),
                MatrixLayout::ColMajor => (self.m * self.k, 1, self.m),
            },
            Ident::Rhs => match self.rhs_layout {
                MatrixLayout::RowMajor => (self.k * self.n, self.n, 1),
                MatrixLayout::ColMajor => (self.k * self.n, 1, self.k),
            },
            Ident::Out => (self.m * self.n, self.n, 1),
        };

        strides.push(y);
        strides.push(x);

        if self.batches.len() > 0 {
            strides.push(last_batch);

            for b in self.batches.iter().rev().take(self.batches.len() - 1) {
                strides.push(last_batch * b)
            }
        }

        strides.into_iter().rev().collect()
    }

    /// Returns the total number of batches
    pub(crate) fn num_batches(&self) -> usize {
        self.batches.iter().map(|&x| x).product()
    }

    /// Asserts that the problem can be solved with the given batch matmul configs
    ///
    /// # Panics:
    ///
    ///  - If dimensions of the problem are larger than allowed by the config
    ///  - If line sizes do not divide well the dimension in which they are aligned
    pub(crate) fn check_config<B: BmmConfig>(&self, config: &B) {
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
