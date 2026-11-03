//! Where a stage keeps each line of a block row ([`RowArrangement`]), swizzled or not.

pub(crate) mod base;
pub(crate) mod swizzle;

pub(crate) use base::{LineBytes, RowArrangement};
pub(crate) use swizzle::{ChunkSwizzle, swizzled_line};
