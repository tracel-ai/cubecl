//! Compilation for a runtime: the one path every backend loads a kernel by,
//! the two halves a backend contributes to it, the persistent stores and the
//! in-memory cache in front of them, and what the environment records of a
//! compilation. The [`Compiler`](cubecl_runtime::compiler::Compiler) contract
//! itself lives in `cubecl-runtime` and is re-exported here.

mod base;
mod cache_key;
mod loader;
mod namespace_root;
mod record;
mod store;
mod target;
mod work_launch;

pub use base::*;
pub use cache_key::*;
pub use loader::*;
pub use namespace_root::*;
pub use record::*;
pub use store::*;
pub use target::*;
pub use work_launch::*;
