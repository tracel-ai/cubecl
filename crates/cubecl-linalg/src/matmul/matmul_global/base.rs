use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::Matmul;

use super::{GmmConfig, Loader, Unloader};

#[cube]
/// Execute a matmul over a block, accumulating for arbitrary k-dim, using one Cube.
pub trait GlobalMatmul<
    EG: Numeric,
    ES: Numeric,
    Lhs: Loader<EG, ES>,
    Rhs: Loader<EG, ES>,
    Out: Unloader<EG>,
>: 'static + Send + Sync + Matmul<EG, EG, Config: GmmConfig>
{
    fn execute(
        lhs_loader: Lhs,
        rhs_loader: Rhs,
        out_writer: Out,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );
}
