use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
fn array_variable(x: u32, y: u32) {
    let _array = [x, y];
}

fn main() {}
