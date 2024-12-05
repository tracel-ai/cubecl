use cubecl_core as cubecl;
use cubecl_core::prelude::*;


#[cube(launch_unchecked)]
pub fn reduce<In: CubeType, Out: CubeType>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<N>,
    axis: u32,
)
