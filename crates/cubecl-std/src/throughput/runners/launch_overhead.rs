use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch_unchecked)]
pub fn launch_overhead<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    output: &mut [Vector<I, N>],
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS == 0 {
        output[0] = input[0];
    }
}
