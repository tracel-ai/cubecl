use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
fn return_value(x: u32, y: u32) -> u32 {
    if x == y {
        return x;
    }

    y
}

fn main() {}
