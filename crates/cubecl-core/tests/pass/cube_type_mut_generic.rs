use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, CubeTypeMut, Clone, Copy)]
pub struct Point<F: Float> {
    pub x: F,
    pub y: F,
}

#[cube]
fn select<F: Float>(cond: bool, a: Point<F>, b: Point<F>) -> Point<F> {
    if cond { a } else { b }
}

fn main() {}
