//! Reports rendered as text: `Text(&report)` implements `Display` for every
//! report the [command line](crate::command) prints.

mod autotune;
mod base;
mod diff;
mod kernels;
mod key;
mod summary;
mod table;
mod timeline;

pub use base::*;
pub use key::KeyText;
