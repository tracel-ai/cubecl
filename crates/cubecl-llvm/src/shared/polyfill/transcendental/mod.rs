//! Transcendentals as polynomials, for a target with no vector math library behind its
//! vector intrinsics: `llvm.exp` on a vector reaches codegen with nothing but a declare
//! behind it, and every lane becomes a libm call that a dozen fused multiply-adds beat.

mod base;
mod exponential;
mod hyperbolic;
mod logarithm;
mod trigonometry;

pub(super) use exponential::exp;
pub(super) use hyperbolic::tanh;
pub(super) use logarithm::ln;
pub(super) use trigonometry::{cos, sin};
