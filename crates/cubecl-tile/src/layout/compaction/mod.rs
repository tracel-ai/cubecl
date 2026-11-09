//! The smallest physical box a gathered operand's sub-tile is staged in ([`Compaction`]), and its
//! in-kernel steps.

pub(crate) mod base;
pub(crate) mod step;

pub(crate) use step::*;

pub use base::Compaction;
