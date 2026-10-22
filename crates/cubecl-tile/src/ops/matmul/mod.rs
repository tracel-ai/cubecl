//! The matmul reading of a [`Tile`](crate::Tile): `c.mma(a, b)` treats the trailing two axes as
//! the `row × col` matrix, leading axes as a batch, and contracts `K`.

pub(crate) mod base;
pub(crate) mod descent;
pub(crate) mod instruction;
pub(crate) mod leaf;
pub(crate) mod mma_io;
pub(crate) mod register_block;

pub(crate) use descent::Descent;
pub(crate) use leaf::*;

pub use mma_io::{LoadMethod, MmaIo, StoreMethod};
