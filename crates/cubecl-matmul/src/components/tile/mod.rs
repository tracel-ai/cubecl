//! Matrix multiplication on register- or shared-memory tiles.
//! Optimized for fixed shapes and low-level compute strategies.

pub mod accelerated;
pub mod io;
pub mod mma;
pub mod plane_vec_mat_inner_product;
pub mod register;

mod base;
mod tile_data;

pub use base::*;
pub use tile_data::*;
