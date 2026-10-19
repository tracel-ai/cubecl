// Cube functions index their slices and swap by hand, since they can't use
// iterators or `mem::swap`.
#![allow(clippy::needless_range_loop, clippy::manual_swap)]

use cubecl_runtime::runtime::Runtime;

use crate::{self as cubecl, prelude::*};

#[cube]
fn double(x: u32) -> u32 {
    x * 2
}

#[cube]
fn multiply_add(a: i32, b: i32, c: i32) -> i32 {
    a * b + c
}

#[cube]
fn scale<F: Float>(x: F, #[comptime] factor: u32) -> F {
    x * F::cast_from(factor)
}

#[cube]
fn truncate_add<F: Float, I: Int>(x: F, y: I) -> I {
    I::cast_from(x) + y
}

#[cube]
fn answer() -> u32 {
    42
}

#[cube]
fn div_rem(a: u32, b: u32) -> (u32, u32) {
    (a / b, a % b)
}

#[cube]
fn split<F: Float>(x: F) -> (F, i32, u32) {
    let whole = F::floor(x);
    (x - whole, i32::cast_from(whole), u32::cast_from(whole) * 2)
}

#[cube]
fn wrap(x: i32) -> (i32,) {
    (x + 1,)
}

#[derive(CubeType, CubeLaunch, CubeHarness, Clone, Copy, Debug, PartialEq)]
pub struct Point<F: Float> {
    x: F,
    y: F,
}

#[derive(CubeType, CubeLaunch, CubeHarness, Clone, Copy, Debug, PartialEq)]
pub struct Segment<F: Float> {
    start: Point<F>,
    end: Point<F>,
    id: u32,
}

#[derive(CubeType, CubeLaunch, CubeHarness, Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    a: u32,
    b: i32,
}

#[cube(harness)]
fn swap<F: Float>(p: Point<F>) -> Point<F> {
    Point::<F> { x: p.y, y: p.x }
}

#[cube(harness)]
fn midpoint<F: Float>(segment: Segment<F>) -> (Point<F>, u32) {
    let x = (segment.start.x + segment.end.x) / F::new(2.0);
    let y = (segment.start.y + segment.end.y) / F::new(2.0);
    (Point::<F> { x, y }, segment.id)
}

#[cube(harness)]
fn translate<F: Float>(p: Point<F>, dx: F, #[comptime] times: u32) -> Point<F> {
    Point::<F> {
        x: p.x + dx * F::cast_from(times),
        y: p.y,
    }
}

#[cube(harness)]
fn flip(pair: Pair) -> (Pair, i32) {
    let flipped = Pair {
        a: pair.a + 1,
        b: -pair.b,
    };
    (flipped, i32::cast_from(pair.a) + pair.b)
}

#[cube]
fn nested(x: u32) -> (u32, (u32, i32)) {
    (x, (x + 1, -1))
}

#[cube]
fn sum(xs: &[u32]) -> u32 {
    let mut total = 0u32;
    for i in 0..xs.len() {
        total += xs[i];
    }
    total
}

#[cube]
fn dot<F: Float>(lhs: &[F], rhs: &[F]) -> F {
    let mut total = F::new(0.0);
    for i in 0..lhs.len() {
        total += lhs[i] * rhs[i];
    }
    total
}

#[cube]
fn fill(start: i32, out: &mut [i32]) {
    for i in 0..out.len() {
        out[i] = start + i32::cast_from(i);
    }
}

#[cube]
fn scale_into<F: Float>(xs: &[F], factor: F, out: &mut [F]) -> u32 {
    for i in 0..xs.len() {
        out[i] = xs[i] * factor;
    }
    u32::cast_from(xs.len())
}

#[cube]
fn swap_firsts(lhs: &mut [u32], rhs: &mut [u32]) {
    let first = lhs[0];
    lhs[0] = rhs[0];
    rhs[0] = first;
}

pub fn test_harness_scalar<R: Runtime>() {
    assert_eq!(double::harness::<R>(1), 2);
    assert_eq!(multiply_add::harness::<R>(-3, 4, 5), -7);
}

/// A function in a block isn't reachable by path from its own module, so the
/// harness must not name it through `super`.
pub fn test_harness_block_scoped<R: Runtime>() {
    #[cube]
    fn negate(x: i32) -> i32 {
        -x
    }

    assert_eq!(negate::harness::<R>(3), -3);
}

/// The number of inputs and outputs don't have to match, and the outputs of a
/// tuple don't have to share a type.
pub fn test_harness_outputs<R: Runtime>() {
    assert_eq!(answer::harness::<R>(), 42);
    assert_eq!(div_rem::harness::<R>(17, 5), (3, 2));
    assert_eq!(split::harness::<R, f32>(3.25), (0.25, 3, 6));
    assert_eq!(wrap::harness::<R>(-8), (-7,));
}

/// Tuples nest, since every value is read back through `HarnessOutput`.
pub fn test_harness_nested<R: Runtime>() {
    assert_eq!(nested::harness::<R>(4), (4, (5, -1)));
}

/// Cube types cross both ways, nested in each other and in tuples.
pub fn test_harness_cube_type<R: Runtime>() {
    assert_eq!(
        swap::harness::<R, f32>(Point { x: 1.0, y: 2.0 }),
        Point { x: 2.0, y: 1.0 }
    );

    let segment = Segment {
        start: Point { x: 0.0, y: 2.0 },
        end: Point { x: 4.0, y: 6.0 },
        id: 7,
    };
    assert_eq!(
        midpoint::harness::<R, f32>(segment),
        (Point { x: 2.0, y: 4.0 }, 7)
    );

    assert_eq!(
        translate::harness::<R, f32>(Point { x: 1.0, y: 5.0 }, 0.5, 4),
        Point { x: 3.0, y: 5.0 }
    );

    assert_eq!(
        flip::harness::<R>(Pair { a: 2, b: 3 }),
        (Pair { a: 3, b: -3 }, 5)
    );
}

/// A host slice goes up as a buffer of its length, and a mutable one comes
/// back written, like the function ran on the host.
pub fn test_harness_slice<R: Runtime>() {
    assert_eq!(sum::harness::<R>(&[1, 2, 3, 4]), 10);
    assert_eq!(
        dot::harness::<R, f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
        32.0
    );

    let mut out = [0; 4];
    fill::harness::<R>(7, &mut out);
    assert_eq!(out, [7, 8, 9, 10]);

    let mut out = [0.0; 3];
    assert_eq!(
        scale_into::harness::<R, f32>(&[1.0, -2.0, 0.5], 2.0, &mut out),
        3
    );
    assert_eq!(out, [2.0, -4.0, 1.0]);

    let (mut lhs, mut rhs) = ([1, 2], [3, 4]);
    swap_firsts::harness::<R>(&mut lhs, &mut rhs);
    assert_eq!((lhs, rhs), ([3, 2], [1, 4]));

    assert_eq!(sum::harness::<R>(&[]), 0);
    fill::harness::<R>(7, &mut []);
}

pub fn test_harness_generic<R: Runtime>() {
    assert_eq!(scale::harness::<R, f32>(1.5, 4), 6.0);
    assert_eq!(truncate_add::harness::<R, f32, i32>(2.75, 3), 5);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_harness {
    () => {
        mod harness {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn harness_scalar() {
                cubecl_core::runtime_tests::harness::test_harness_scalar::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_block_scoped() {
                cubecl_core::runtime_tests::harness::test_harness_block_scoped::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_outputs() {
                cubecl_core::runtime_tests::harness::test_harness_outputs::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_nested() {
                cubecl_core::runtime_tests::harness::test_harness_nested::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_cube_type() {
                cubecl_core::runtime_tests::harness::test_harness_cube_type::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_slice() {
                cubecl_core::runtime_tests::harness::test_harness_slice::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_generic() {
                cubecl_core::runtime_tests::harness::test_harness_generic::<TestRuntime>();
            }
        }
    };
}
