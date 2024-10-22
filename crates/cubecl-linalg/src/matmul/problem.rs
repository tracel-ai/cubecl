use super::{matmul_batch::BmmConfig, matrix::MatrixLayout};

#[derive(Copy, Clone)]
pub struct MatmulProblem {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub b: u32,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,
}
impl MatmulProblem {
    pub(crate) fn check_config<B: BmmConfig>(&self, config: &B) {
        assert!(
            self.m <= config.max_m(),
            "Problem has m={} but these configs can only have m<={}",
            self.m,
            config.max_m()
        );
        assert!(
            self.n <= config.max_n(),
            "Problem has n={} but these configs can only have n<={}",
            self.n,
            config.max_n()
        );
        assert!(
            self.b <= config.max_batches(),
            "Problem has {} batches but these configs can only have batches<={}",
            self.b,
            config.max_batches()
        );

        // Lhs
        assert!(match self.lhs_layout {
            MatrixLayout::RowMajor => self.k % self.lhs_line_size as u32 == 0,
            MatrixLayout::ColMajor => self.m % self.lhs_line_size as u32 == 0,
        });

        // Rhs
        assert!(match self.rhs_layout {
            MatrixLayout::RowMajor => self.n % self.rhs_line_size as u32 == 0,
            MatrixLayout::ColMajor => self.k % self.rhs_line_size as u32 == 0,
        });

        // Out
        assert!(self.n % self.out_line_size as u32 == 0);
    }
}
