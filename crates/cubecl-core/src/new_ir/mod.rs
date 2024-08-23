mod branch;
mod expression;
mod literal;
mod operators;
mod statement;
mod types;

use std::num::NonZero;

pub use branch::*;
pub use expression::*;
pub use literal::*;
pub use operators::*;
pub use statement::*;
pub use types::*;

pub use crate::ir::Elem;
pub use cubecl_common::operator::Operator;

pub fn assert_valid_type<T: KernelArg>() {}

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
