use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct TmpSlice<N: Numeric> {
    _n: Array<N>,
}

#[derive(CubeType)]
pub struct TmpSliceMut<N: Numeric> {
    _n: Array<N>,
}
