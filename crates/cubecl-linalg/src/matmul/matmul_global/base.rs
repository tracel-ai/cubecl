use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::Matmul;

use super::{GmmConfig, Loader, Unloader};

#[cube]
pub trait GlobalMatmul<
    EG: Numeric,
    ES: Numeric,
    Lhs: Loader<EG, ES, G>,
    Rhs: Loader<EG, ES, G>,
    Out: Unloader<EG, G>,
    G: GmmConfig,
>: 'static + Send + Sync + Matmul<EG, EG, Config = G>
{
    fn execute(
        lhs_loader: Lhs,
        rhs_loader: Rhs,
        out_writer: Out,
        x_offset: u32,
        y_offset: u32,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );
}
