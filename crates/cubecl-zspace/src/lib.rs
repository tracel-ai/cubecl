//! # Common `ZSpace` Utilities for `CubeCL`
//!
//! This is a new/experimental module.
//!
//! The goal will be to unify:
//! - shape/stride construction/validation.
//! - stride map view transformations.
//! - common Shape types.
//! - common Shape/Size/Reshape utility traits.
//!
//! The intention is to publish this as a stand-alone `zspace` module,
//! with no direct tie to `cubecl`; once it is more polished.

#![no_std]

extern crate alloc;

pub mod errors;
pub mod indexing;
pub mod striding;

pub(crate) const INLINE_DIMS: usize = 5;

pub mod metadata;
mod shape;
mod strides;

/// Reexport to avoid annoying rust-analyzer bug where it imports the module instead of the macro
pub use shape::*;
pub use strides::*;

/// Reexport for use in macros
pub use smallvec::{SmallVec, smallvec};
