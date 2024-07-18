mod base;
mod block_loop;
mod compute_loop;
pub(crate) mod config;
mod launch;
mod load_shared_memory;
mod outer_product;
mod tile;
mod write_output;

pub use launch::matmul_tiling_2d_cube;

#[cfg(feature = "export_tests")]
pub use {
    compute_loop::tests as compute_loop_tests,
    load_shared_memory::tests as load_shared_memory_tests,
    outer_product::tests as outer_product_tests, write_output::tests as write_output_tests,
};
