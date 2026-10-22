//! The [`Arg`] builder, which binds a launched tensor to an operand: its labels and where its
//! bounds-check lands.

pub(crate) mod base;
pub(crate) mod boundaries;
pub(crate) mod derivation;
pub(crate) mod labels;

pub(crate) use base::rebound_tensor;
pub use base::{Arg, Bound, Labelled, Unbound, Unlabelled};
