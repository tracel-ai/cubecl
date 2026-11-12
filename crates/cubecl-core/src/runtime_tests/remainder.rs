//! Folds of a remainder or a high multiply by a constant that is one (or minus one).
//!
//! Each kernel computes the operation twice per cube. The first slot uses the constant as a
//! literal, so the operation is built with a constant operand and folds through its
//! `const_eval!` rules. The second slot reaches the same constant only through a `select`
//! between constants, so the operand is no constant when the operation is built and the
//! operation folds once the `select` does.
//!
//! The kernels compute in `I` and write `O`, so an index-typed (`usize`) operation is checked
//! through a `u32` buffer. The output starts filled with a sentinel, so a slot the kernel never
//! wrote fails the test.

use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;

/// Cubes launched, so the dividends are the cube positions `0..CUBES` minus an offset.
pub const CUBES: u32 = 4;

/// Slots written by each cube: the literal operand, then the folded one.
const SLOTS: usize = 2;

#[cube(launch_unchecked)]
pub fn kernel_remainder_by<I: Int, O: Int>(
    output: &mut [O],
    #[comptime] divisor: i32,
    #[comptime] offset: i64,
) {
    let at = CUBE_POS_X as usize * SLOTS;
    let dividend = I::cast_from(CUBE_POS_X) - I::new(offset);
    let runtime = I::cast_from(divisor.runtime());
    let folded = select(runtime == I::new(0), I::new(divisor as i64), runtime);
    output[at] = O::cast_from(dividend % I::new(divisor as i64));
    output[at + 1] = O::cast_from(dividend % folded);
}

#[cube(launch_unchecked)]
pub fn kernel_mul_hi_by_one<I: Int + MulHi, O: Int>(output: &mut [O], #[comptime] offset: i64) {
    let at = CUBE_POS_X as usize * SLOTS;
    let factor = I::cast_from(CUBE_POS_X) - I::new(offset);
    let runtime = I::cast_from(1i32.runtime());
    let folded = select(runtime == I::new(0), I::new(1), runtime);
    output[at] = O::cast_from(I::mul_hi(factor, I::new(1)));
    output[at + 1] = O::cast_from(I::mul_hi(folded, factor));
}

/// Launches `CUBES` single-unit cubes over an output filled with `O::max_value()`.
fn launch_over_sentinel<O: Int + CubeElement>(
    client: &Client,
    launch: impl FnOnce(BufferArg),
) -> Vec<O> {
    let len = CUBES as usize * SLOTS;
    let sentinel = alloc::vec![O::max_value(); len];
    let output = client.create_from_slice(O::as_bytes(&sentinel));
    launch(unsafe { BufferArg::from_raw_parts(output.clone(), len) });
    O::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

/// Every cube's remainders by `divisor`, literal and folded, of `CUBE_POS_X - offset`.
pub fn remainders_by<I: Int, O: Int + CubeElement>(
    client: &Client,
    divisor: i32,
    offset: i64,
) -> Vec<O> {
    launch_over_sentinel::<O>(client, |output| unsafe {
        kernel_remainder_by::launch_unchecked::<I, O>(
            client,
            CubeCount::Static(CUBES, 1, 1),
            CubeDim::new_1d(1),
            output,
            divisor,
            offset,
        )
    })
}

/// Every cube's high halves of `(CUBE_POS_X - offset) * 1`, literal and folded.
pub fn mul_his_by_one<I: Int + MulHi, O: Int + CubeElement>(
    client: &Client,
    offset: i64,
) -> Vec<O> {
    launch_over_sentinel::<O>(client, |output| unsafe {
        kernel_mul_hi_by_one::launch_unchecked::<I, O>(
            client,
            CubeCount::Static(CUBES, 1, 1),
            CubeDim::new_1d(1),
            output,
            offset,
        )
    })
}

/// The value every slot of every cube should hold.
fn every_slot<O: Clone>(value: impl Fn(i64) -> O, offset: i64) -> Vec<O> {
    (0..CUBES as i64)
        .flat_map(|cube| core::iter::repeat_n(value(cube - offset), SLOTS))
        .collect()
}

pub fn test_remainder_by_one(client: Client) {
    let zeros = |_| 0u32;
    assert_eq!(
        remainders_by::<u32, u32>(&client, 1, 0),
        every_slot(zeros, 0)
    );
    assert_eq!(
        remainders_by::<usize, u32>(&client, 1, 0),
        every_slot(zeros, 0)
    );
    assert_eq!(
        remainders_by::<i32, i32>(&client, 1, 2),
        every_slot(|_| 0i32, 2)
    );
}

pub fn test_remainder_by_minus_one(client: Client) {
    assert_eq!(
        remainders_by::<i32, i32>(&client, -1, 2),
        every_slot(|_| 0i32, 2)
    );
}

pub fn test_mul_hi_by_one(client: Client) {
    let zeros = |_| 0u32;
    assert_eq!(mul_his_by_one::<u32, u32>(&client, 0), every_slot(zeros, 0));
    assert_eq!(
        mul_his_by_one::<usize, u32>(&client, 0),
        every_slot(zeros, 0)
    );
    // The high half of a signed `x * 1` is the sign extension of `x`.
    let sign = |x: i64| if x < 0 { -1i32 } else { 0 };
    assert_eq!(mul_his_by_one::<i32, i32>(&client, 2), every_slot(sign, 2));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_remainder {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_remainder_by_one_is_zero() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::remainder::test_remainder_by_one(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_remainder_by_minus_one_is_zero() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::remainder::test_remainder_by_minus_one(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_mul_hi_by_one_is_the_sign() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::remainder::test_mul_hi_by_one(client);
        }
    };
}
