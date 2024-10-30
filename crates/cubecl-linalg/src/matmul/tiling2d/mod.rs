pub(crate) mod base;
mod block_loop;
pub(crate) mod compute_loop;
pub(crate) mod config;
mod launch;
pub(crate) mod load_shared_memory;
pub(crate) mod outer_product;
pub(crate) mod tile;
pub(crate) mod write_output;

pub use launch::matmul_tiling_2d as launch;
pub use launch::matmul_tiling_2d_ref as launch_ref;
