//! Reads a cubecl environment file back.
//!
//! An environment is one database: compiled kernels, autotune results, and
//! whatever an application stores beside them. Everything the runtime writes
//! there is written to be served, and this crate is the other reader — the one
//! that answers *what does this file hold, and how was each autotune key
//! decided?*
//!
//! Three layers, each usable without the one above:
//!
//! - [`Inspector`] opens a file read-only and produces [reports](report): plain
//!   serializable values, which is what tests pin.
//! - [`view`] renders a report as text.
//! - [`command`] is the command line both front doors share: the
//!   `cubecl-inspect` binary, and any application that wraps it and resolves
//!   which file to open its own way.
//! - `tui` (feature `tui`) browses the same reports in a terminal, re-read
//!   as the file changes.

mod error;
mod inspector;

pub mod command;
pub mod report;
#[cfg(feature = "tui")]
pub mod tui;
pub mod view;

pub use error::InspectError;
pub use inspector::Inspector;
