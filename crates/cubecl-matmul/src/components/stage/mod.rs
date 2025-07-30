// ! Performs tiled matrix multiplication using shared memory.
// ! Manages unit/plane coordination

mod matmul;

mod base;
mod event_listener;
mod memory;

pub use base::*;
pub use event_listener::*;
pub use matmul::*;
pub use memory::*;
