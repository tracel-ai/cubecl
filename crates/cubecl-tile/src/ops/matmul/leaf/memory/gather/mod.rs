//! The N-D nest, for operands whose contracted axes are gathered rather than lined.

pub(crate) mod base;
pub(crate) mod coords;
pub(crate) mod nd;
pub(crate) mod separable;

pub(crate) use base::*;
