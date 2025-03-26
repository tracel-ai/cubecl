mod base;
mod selector;

pub mod double_buffering;
pub mod double_buffering_barrier;
pub mod simple;
pub mod simple_barrier;
pub mod simple_pipelined;
pub mod simple_tma;
pub mod specialized;

pub use base::Algorithm;
pub use selector::*;
