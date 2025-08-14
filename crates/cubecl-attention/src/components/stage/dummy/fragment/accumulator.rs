use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatmulPrecision;
use cubecl_matmul::components::tile::TileMatmul;

#[derive(CubeType)]
pub struct AccumulatorFragment<MP: MatmulPrecision, VTM: TileMatmul<MP>> {
    pub fragment: VTM::Accumulator,
}

#[cube]
impl<MP: MatmulPrecision, VTM: TileMatmul<MP>> AccumulatorFragment<MP, VTM> {
    pub fn new(#[comptime] config: VTM::Config) -> AccumulatorFragment<MP, VTM> {
        let mut fragment = VTM::allocate_accumulator(config);
        VTM::zero_accumulator(&mut fragment, config);
        AccumulatorFragment::<MP, VTM> { fragment }
    }
}
