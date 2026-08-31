#[macro_use]
extern crate derive_new;

extern crate alloc;

pub mod amdgpu;
pub mod cpu;
pub mod shared;
pub mod target;

pub use cpu::jit::data::{PlironData, SharedData};
pub use cpu::jit::engine::{KernelRequirements, PlironEngine};
pub use cpu::shared_memory::SharedMemories;
pub use shared::{AmdGpuModule, PlironArtifact, PlironCompiler, PlironOptions};
pub use target::LlvmTarget;
