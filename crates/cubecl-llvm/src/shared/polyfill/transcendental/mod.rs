//! Polynomial approximations for vector math.

pub(crate) mod base;
mod exponential;
mod hyperbolic;
mod logarithm;
mod trigonometry;

pub use exponential::exp;
pub use hyperbolic::tanh;
pub use logarithm::ln;
pub use trigonometry::{cos, sin};
