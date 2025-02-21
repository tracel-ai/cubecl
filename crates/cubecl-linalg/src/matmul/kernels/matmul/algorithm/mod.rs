mod base;
mod selector;

pub mod double_buffering;
pub mod simple;
pub mod simple_barrier_strided;
pub mod simple_pipelined;
pub mod simple_strided;
pub mod specialized;

pub use base::Algorithm;
pub use selector::*;
