#![allow(unused_mut)]

use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
fn index(first: usize, second: usize, dims: (usize, usize)) -> usize {
    dims.0 * second + first
}

#[cube]
fn index_comptime(first: usize, second: usize, #[comptime] dims: (usize, usize)) -> usize {
    dims.0 * second + first
}

#[cube]
fn index_nested(first: usize, dims: ((usize, usize), u32)) -> usize {
    (dims.0).0 * first + usize::cast_from(dims.1)
}

#[cube]
fn tuple_literal_mut(a: usize, b: usize) -> usize {
    let mut dims = (2usize, 2usize);
    index(a, b, dims)
}

#[cube]
fn tuple_literal(a: usize, b: usize) -> usize {
    let dims = (2usize, 2usize);
    index(a, b, dims)
}

#[cube]
fn tuple_literal_inline(a: usize, b: usize) -> usize {
    index(a, b, (2usize, 2usize))
}

#[cube]
fn tuple_literal_nested(a: usize) -> usize {
    let mut dims = ((2usize, 3usize), 4u32);
    index_nested(a, dims)
}

#[cube]
fn tuple_literal_comptime(a: usize, b: usize) -> usize {
    let dims = (2usize, 2usize);
    index_comptime(a, b, dims)
}

#[cube]
fn tuple_literal_inline_comptime(a: usize, b: usize) -> usize {
    index_comptime(a, b, (2usize, 2usize))
}

#[cube]
fn tuple_of_runtime_vars(a: usize, b: usize) -> usize {
    let mut d = 2usize;
    index(a, b, (d, d))
}

#[cube]
fn tuple_literal_typed(a: usize, b: usize) -> usize {
    let dims: (usize, usize) = (2, 2);
    index(a, b, dims)
}

#[cube]
fn tuple_literal_rebound(a: usize, b: usize) -> usize {
    let dims = (2usize, 2usize);
    let other = dims;
    index(a, b, other)
}

#[cube]
fn comptime_param_tuple(a: usize, b: usize, #[comptime] dims: (usize, usize)) -> usize {
    index(a, b, dims)
}

#[cube]
fn comptime_nested_param_tuple(a: usize, #[comptime] dims: ((usize, usize), u32)) -> usize {
    index_nested(a, dims)
}

fn main() {}
