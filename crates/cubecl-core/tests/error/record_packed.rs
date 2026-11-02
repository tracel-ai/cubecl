use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
#[repr(C, packed)]
struct Packed {
    a: f32,
    b: f64,
}

fn main() {}
