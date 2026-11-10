use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
#[repr(C)]
struct WithComptime {
    a: f32,
    #[cube(comptime)]
    n: u32,
}

fn main() {}
