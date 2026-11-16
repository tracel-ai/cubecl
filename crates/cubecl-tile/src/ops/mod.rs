//! The verbs a client runs over tiles: [`matmul`] (`mma`), `product`, `reduce`, [`softmax`] and
//! `relu`.

pub(crate) mod contraction;
pub mod matmul;
pub(crate) mod product;
pub(crate) mod reduce;
pub(crate) mod relu;
pub mod softmax;

pub(crate) use matmul::*;
// product and relu add `Tile` impls only; nothing to re-export.
