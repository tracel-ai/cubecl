// ! Performs tiled matrix multiplication using shared memory.
// ! Manages unit/plane coordination

mod matmul;

mod base;
mod event_listener;
mod layout;
mod reader;
mod stage_memory;

pub use base::*;
pub use event_listener::*;
pub use layout::*;
pub use matmul::*;
pub use reader::*;
pub use stage_memory::StageMemory;
