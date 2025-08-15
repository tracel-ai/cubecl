use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::tile::ValueMatmul;

#[derive(CubeType)]
pub struct AccumulatorFragment<AP: AttentionPrecision, VM: ValueMatmul<AP>> {
    pub fragment: VM::Accumulator,
}

#[cube]
impl<AP: AttentionPrecision, VM: ValueMatmul<AP>> AccumulatorFragment<AP, VM> {
    pub fn new(#[comptime] config: VM::Config) -> AccumulatorFragment<AP, VM> {
        let mut fragment = VM::allocate_accumulator(config);
        VM::zero_accumulator(&mut fragment, config);
        AccumulatorFragment::<AP, VM> { fragment }
    }
}
