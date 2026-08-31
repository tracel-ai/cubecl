//! Transcendentals as polynomials, for a target with no vector math library behind its
//! vector intrinsics: `llvm.exp` on a vector reaches codegen with nothing but a declare
//! behind it, and every lane becomes a libm call that a dozen fused multiply-adds beat.

pub(crate) mod base;
mod exponential;
mod hyperbolic;
mod logarithm;
mod trigonometry;

pub use exponential::exp;
pub use hyperbolic::tanh;
pub use logarithm::ln;
pub use trigonometry::{cos, sin};
