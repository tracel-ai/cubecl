#[macro_use]
extern crate derive_new;

extern crate alloc;

pub mod shared;

pub use shared::jit::data::{PlironData, SharedData};
pub use shared::jit::engine::{KernelRequirements, PlironEngine};
pub use shared::{PlironCompiler, PlironOptions};
