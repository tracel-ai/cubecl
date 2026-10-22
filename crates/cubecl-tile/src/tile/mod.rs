//! One operand's data in the kernel: the [`Tile`], its kinds, packing and accumulators.

pub(crate) mod accumulator;
pub(crate) mod base;
pub(crate) mod element;
pub(crate) mod factor;
pub(crate) mod held;
pub(crate) mod index;
pub(crate) mod kind;
pub(crate) mod packing;
pub(crate) mod placement;
pub(crate) mod slices;

pub(crate) use accumulator::*;
pub(crate) use element::*;
pub(crate) use factor::*;
pub(crate) use index::*;
pub(crate) use kind::*;
pub(crate) use packing::*;
pub(crate) use placement::*;
