//! Defines the loading strategy used to bring data into shared memory.

mod base;
pub use base::*;

pub mod async_full_cooperative;
pub mod async_full_cyclic;
pub mod async_full_maximize_slice_length;
pub mod async_full_maximize_unit_count;

pub mod async_partial_maximize_slice_length;

pub mod sync_full_cyclic;
pub mod sync_full_ordered;
pub mod sync_full_strided;
pub mod sync_full_tilewise;
pub mod sync_partial_cyclic;

pub mod sync_partial_tilewise;
