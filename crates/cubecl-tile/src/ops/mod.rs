//! The verbs a client runs over tiles: [`matmul`] (`mma`), `product`, [`softmax`] and `rows`.

pub mod matmul;
pub(crate) mod product;
pub(crate) mod reduce;
pub(crate) mod rows;
pub mod softmax;
pub(crate) mod team;

pub(crate) use matmul::*;
// product and rows add `Tile` impls only; nothing to re-export.

pub use team::TeamUnit;
