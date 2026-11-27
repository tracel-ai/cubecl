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

#[cube(launch)]
fn tuple_field_kernel(output: &mut [u32], tuple: (u32, u32)) {
    output[0] = tuple.0 + tuple.1;
}

#[cube(launch)]
fn nested_tuple_destructure_kernel(
    output: &mut [u32],
    tuple: (u32, u32, (u32, u32, (u32, u32))),
) {
    let (a, b, (c, d, (e, f))) = tuple;
    output[0] = a + b + c + d + e + f;
}

#[cube(launch)]
fn sibling_nested_tuple_destructure_kernel(
    output: &mut [u32],
    tuple: ((u32, u32), u32, (u32, (u32, u32))),
) {
    output[0] = sibling_nested_tuple_destructure(tuple);
}

fn main() {}
