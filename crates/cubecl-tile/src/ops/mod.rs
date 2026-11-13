//! The verbs a client runs over tiles: [`matmul`] (`mma`), `product`, `reduce` and [`softmax`].

pub(crate) mod contraction;
pub mod matmul;
pub(crate) mod product;
pub(crate) mod reduce;
pub mod softmax;

pub(crate) use matmul::*;
// product adds `Tile` impls only; nothing to re-export.
