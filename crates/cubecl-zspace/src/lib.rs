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
extern crate alloc;

pub mod errors;
pub mod indexing;
pub mod metadata_next;
pub mod striding;
