use cubecl_core::prelude::*;

#[cube]
fn range(x: UInt, y: UInt) -> UInt {
    if x == y {
        return x;
    }

    y
}

fn main() {}
