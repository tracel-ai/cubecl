use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Dimensions {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

#[cube]
pub(crate) fn get_dims<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>) -> Dimensions {
    let rank = lhs.rank();
    let first_dim = rank - 2;
    let second_dim = rank - 1;
    let m = lhs.shape(first_dim);
    let k = lhs.shape(second_dim);
    let n = rhs.shape(second_dim);

    Dimensions { m, k, n }
}
