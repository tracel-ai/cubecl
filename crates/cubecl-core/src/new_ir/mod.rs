mod array;
mod branch;
mod expression;
mod globals;
mod launch;
mod operators;
mod option;
mod statement;
mod subcube;
mod tensor;
mod types;

pub mod compute;
pub mod element;

use std::num::NonZero;

pub use array::*;
pub use branch::*;
pub use compute::*;
pub use expression::*;
pub use globals::*;
pub use launch::*;
pub use operators::*;
pub use option::*;
pub use statement::*;
pub use subcube::*;
pub use tensor::*;
pub use types::*;

pub use crate::ir::Elem;
pub use cubecl_common::operator::Operator;

pub fn assert_valid_type<T: LaunchArg>() {}

/// Calculate the lergest common vectorization of two optional vectorizations
pub fn largest_common_vectorization(
    left_vec: Option<NonZero<u8>>,
    right_vec: Option<NonZero<u8>>,
) -> Option<NonZero<u8>> {
    match (left_vec, right_vec) {
        (None, Some(right)) => Some(right),
        (Some(left), None) => Some(left),
        (Some(left), Some(right)) => {
            let smaller = left.min(right).get();
            let common = (1..=smaller)
                .rev()
                .find(|divisor| left.get() % divisor == 0 && right.get() % divisor == 0)
                .unwrap_or(1);
            // We know it can't be zero
            Some(unsafe { NonZero::new_unchecked(common) })
        }
        _ => None,
    }
}
