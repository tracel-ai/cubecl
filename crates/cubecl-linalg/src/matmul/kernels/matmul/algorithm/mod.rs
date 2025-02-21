mod base;
mod selector;

pub mod double_buffering;
pub mod simple;
pub mod simple_pipelined;
pub mod simple_barrier;
pub mod specialized;

pub use base::Algorithm;
pub use selector::*;
