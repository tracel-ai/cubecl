//! The reports: plain values an [`Inspector`](crate::Inspector) produces and a
//! [view](crate::view) renders. Nothing here reads a file or prints.

mod activity;
mod autotune;
mod candidates;
mod diff;
mod kernels;
mod key;
mod memory;
mod overview;
mod summary;
mod timeline;

pub use activity::*;
pub use autotune::*;
pub use candidates::*;
pub use diff::*;
pub use kernels::*;
pub use key::{AutotuneTable, KeyId};
// Matches keys across the records and the files; no report carries one.
pub(crate) use key::CanonicalKey;
pub use memory::*;
pub use overview::*;
pub use summary::*;
pub use timeline::*;
