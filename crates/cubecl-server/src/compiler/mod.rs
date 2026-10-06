//! Compilation caching for a runtime: the persistent store and the in-memory
//! cache in front of it, and what the environment records of a compilation.
//! The [`Compiler`] contract itself lives in `cubecl-runtime` and is
//! re-exported here.

mod base;
mod record;

pub use base::*;
pub use record::*;
