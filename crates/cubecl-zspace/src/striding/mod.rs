//! # Stride Utilities

mod layout_builders;
mod layout_validation;

pub use crate::errors::StrideError;
pub use crate::errors::StrideRecord;
pub use layout_builders::row_major_contiguous_strides;
pub use layout_validation::{
    has_contiguous_row_major_strides, has_pitched_row_major_strides,
    try_check_contiguous_row_major_strides, try_check_matching_ranks,
    try_check_pitched_row_major_strides,
};
