use crate::matmul::matmul_global::ReadView;
use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Stage<ES: Numeric>:
    CubeType + Clone + Copy + IntoRuntime + Send + Sync + 'static
{
    type Underlying: CubeType;
    type Config: SmmConfig;

    fn new(#[comptime] ident: Ident, #[comptime] config: Self::Config) -> Self;

    fn fill<EG: Numeric, RV: ReadView<EG, Config = Self::Config>>(
        self_: &mut Self,
        global: &RV,
        #[comptime] ident: Ident,
        #[comptime] config: Self::Config,
    );

    fn get_tile(
        self_: &Self,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: Self::Config,
    ) -> &Slice<'_, Line<ES>>;
}
