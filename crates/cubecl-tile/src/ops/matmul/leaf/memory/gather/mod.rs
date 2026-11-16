//! The N-D nest, for operands whose contracted axes are gathered rather than lined.

pub(crate) mod base;
pub(crate) mod coords;
pub(crate) mod nd;
pub(crate) mod row_vectors;
pub(crate) mod separable;

pub use base::LhsVectorMisfit;
pub(crate) use base::*;
