use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
fn nested_tuple_destructure(tuple: (u32, u32, (u32, u32, (u32, u32)))) -> u32 {
    let (a, b, (c, d, (e, f))) = tuple;
    a + b + c + d + e + f
}

#[cube]
fn sibling_nested_tuple_destructure(
    tuple: ((u32, u32), u32, (u32, (u32, u32))),
) -> u32 {
    let ((a, b), c, (d, (e, f))) = tuple;
    a + b + c + d + e + f
}

fn main() {}
