//! Compilation for a runtime: the one path every backend loads a kernel by,
//! the two halves a backend contributes to it, the persistent stores and the
//! in-memory cache in front of them, and what the environment records of a
//! compilation. The [`Compiler`](cubecl_runtime::compiler::Compiler) contract
//! itself lives in `cubecl-runtime` and is re-exported here.

mod base;
mod loader;
mod record;
mod store;
mod target;

pub use base::*;
pub use loader::*;
pub use record::*;
pub use store::*;
pub use target::*;
