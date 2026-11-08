//! A remainder by a divisor that folds to one is zero: written as a literal, and reached only
//! once a `select` between constants is folded, where the divisor is no constant when the
//! remainder is built.

use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;

/// Cubes launched, so the dividends are the cube positions `0..CUBES`, not only zero.
pub const CUBES: u32 = 4;

#[cube(launch_unchecked)]
pub fn kernel_unsigned_remainder_by_one(output: &mut [u32], #[comptime] divisor: u32) {
    let at = CUBE_POS_X as usize * 2;
    let divisor = divisor.runtime();
    let folded = select(divisor == 0u32, 1u32, divisor);
    output[at] = CUBE_POS_X % 1u32;
    output[at + 1] = CUBE_POS_X % folded;
}

#[cube(launch_unchecked)]
pub fn kernel_signed_remainder_by_one(output: &mut [i32], #[comptime] divisor: i32) {
    let at = CUBE_POS_X as usize * 2;
    let dividend = CUBE_POS_X as i32;
    let divisor = divisor.runtime();
    let folded = select(divisor == 0i32, 1i32, divisor);
    output[at] = dividend % 1i32;
    output[at + 1] = dividend % folded;
}

/// Every cube's remainders by one, literal and folded, unsigned.
pub fn unsigned_remainders_by_one(client: &Client) -> Vec<u32> {
    let output = client.empty(CUBES as usize * 2 * core::mem::size_of::<u32>());
    unsafe {
        kernel_unsigned_remainder_by_one::launch_unchecked(
            client,
            CubeCount::Static(CUBES, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), CUBES as usize * 2),
            1,
        )
    };
    u32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

/// Every cube's remainders by one, literal and folded, signed.
pub fn signed_remainders_by_one(client: &Client) -> Vec<i32> {
    let output = client.empty(CUBES as usize * 2 * core::mem::size_of::<i32>());
    unsafe {
        kernel_signed_remainder_by_one::launch_unchecked(
            client,
            CubeCount::Static(CUBES, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), CUBES as usize * 2),
            1,
        )
    };
    i32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

pub fn test_remainder_by_one(client: Client) {
    let zeros = CUBES as usize * 2;
    assert_eq!(
        unsigned_remainders_by_one(&client),
        alloc::vec![0u32; zeros]
    );
    assert_eq!(signed_remainders_by_one(&client), alloc::vec![0i32; zeros]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_remainder {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_remainder_by_a_folded_one_is_zero() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::remainder::test_remainder_by_one(client);
        }
    };
}
