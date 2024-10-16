use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::BmmConfig;

#[cube]
/// Execute a matmul on a whole tensor
pub trait BatchMatmul<N: Numeric> {
    type Config: BmmConfig;

    fn execute(
        lhs: &Tensor<Line<N>>,
        rhs: &Tensor<Line<N>>,
        out: &mut Tensor<Line<N>>,
        #[comptime] config: Self::Config,
    );
}
