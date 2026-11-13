//! A destination whose stored vectors go through a [`VectorTransform`].

pub(crate) mod base;
pub(crate) mod transform;
pub(crate) mod vectors;

pub use base::{Transformed, TransformedOperand};
pub use transform::{VectorAt, VectorTransform, VectorTransformLaunch};
