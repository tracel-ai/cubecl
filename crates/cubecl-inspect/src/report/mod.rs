//! The reports: plain values an [`Inspector`](crate::Inspector) produces and a
//! [view](crate::view) renders. Nothing here reads a file or prints.

mod autotune;
mod candidates;
mod compiling;
mod diff;
mod kernels;
mod key;
mod memory;
mod records;
mod summary;
mod timeline;

pub use autotune::*;
pub use candidates::*;
pub use compiling::*;
pub use diff::*;
pub use kernels::*;
pub use key::*;
pub use memory::*;
pub use records::*;
pub use summary::*;
pub use timeline::*;
