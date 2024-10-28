#[cfg(feature = "export_tests")]
use super::matrix::Ident;
use super::{matmul_batch::BmmConfig, matrix::MatrixLayout};

#[derive(Clone)]
pub struct MatmulProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub b: Vec<usize>,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,
}

impl MatmulProblem {
    #[cfg(feature = "export_tests")]
    pub(crate) fn tensor_size(&self, ident: Ident) -> usize {
        match ident {
            Ident::Lhs => self.num_batches() * self.m * self.k,
            Ident::Rhs => self.num_batches() * self.k * self.n,
            Ident::Out => self.num_batches() * self.m * self.n,
        }
    }

    #[cfg(feature = "export_tests")]
    pub(crate) fn shape(&self, ident: Ident) -> Vec<usize> {
        self.b
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
    pub(crate) fn strides(&self, ident: Ident) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.b.len() + 2);

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

        if self.b.len() > 0 {
            strides.push(last_batch);

            for b in self.b.iter().rev().take(self.b.len() - 1) {
                strides.push(last_batch * b)
            }
        }

        strides.into_iter().rev().collect()
    }

    pub(crate) fn num_batches(&self) -> usize {
        self.b.iter().map(|&x| x).product()
    }

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
