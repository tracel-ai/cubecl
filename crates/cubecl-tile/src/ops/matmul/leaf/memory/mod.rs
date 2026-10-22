//! The register nest for an accumulator in memory, seeded from and committed to the sink per
//! visit.

pub(crate) mod base;
pub(crate) mod direct;
pub(crate) mod gather;
pub(crate) mod shape;

pub(crate) use base::{contract, contracted_per_step, resolve_nd_coords};
