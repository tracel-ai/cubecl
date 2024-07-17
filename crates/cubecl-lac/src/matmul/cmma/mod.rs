mod base;
mod block_loop;
mod compute_loop;
mod config;
mod launch;
mod load_shared_memory;
mod write_output;

pub use launch::matmul_cmma;

#[cfg(feature = "export_tests")]
pub use {
    compute_loop::tests as cmma_compute_loop_tests,
    // compute_loop_mimic::tests as cmma_compute_loop_mimic_tests,
    load_shared_memory::tests as cmma_load_shared_memory_tests,
    write_output::tests as cmma_write_output_tests,
};
