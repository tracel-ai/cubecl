use cubecl::prelude::*;
use cubecl_core as cubecl;

// Regression for #1887: RuntimeAssign bounds must preserve the equality
// between a field's expanded type and the result of init_mut.
#[derive(CubeType, CubeTypeMut)]
pub struct Point<F: Float> {
    pub x: F,
    pub y: F,
}

#[derive(CubeType, CubeTypeMut)]
pub struct Wrapper<T: CubeType> {
    pub value: T,
}

#[cube]
pub fn select_nested<F: Float>(
    cond: bool,
    a: Wrapper<Point<F>>,
    b: Wrapper<Point<F>>,
) -> Wrapper<Point<F>> {
    if cond { a } else { b }
}

fn main() {}
