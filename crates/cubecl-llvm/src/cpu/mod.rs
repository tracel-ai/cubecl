//! The CPU (host) target.
//!
//! Everything the JIT host needs and the GPU has no use for: the emulation loop that turns one
//! call into a cube launch, the pointer table its resources arrive through, the barrier and
//! atomics that emulation implies, and the JIT that runs the result.

pub mod abi;
pub mod entrypoint;
pub mod jit;
pub mod ordered_atomic;
pub mod shared_memory;
pub mod synchronization;
pub mod to_llvm;
