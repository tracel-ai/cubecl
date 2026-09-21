//! The reports: plain values an [`Inspector`](crate::Inspector) produces and a
//! [view](crate::view) renders. Nothing here reads a file or prints.

mod autotune;
mod candidates;
mod diff;
mod kernels;
mod key;
mod summary;
mod timeline;

pub use autotune::*;
pub use candidates::*;
pub use diff::*;
pub use kernels::*;
pub use key::*;
pub use summary::*;
pub use timeline::*;
