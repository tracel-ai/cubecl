//! Reads a cubecl environment file back.
//!
//! An environment is one database: compiled kernels, autotune results, and
//! whatever an application stores beside them. Everything the runtime writes
//! there is written to be served, and this crate is the other reader — the one
//! that answers *what does this file hold, and how was each autotune key
//! decided?*
//!
//! The layers, each usable without the ones above:
//!
//! - [`EnvironmentName`] resolves `--env` the way the runtime resolves its
//!   own environment, from the configuration above the working directory.
//! - [`Inspector`] opens a file read-only and produces [reports](report): plain
//!   serializable values, which is what tests pin. [`list`] lists the
//!   environment files of some directories, a summary each.
//! - [`view`] renders a report as text.
//! - [`command`] is the command line both front doors share: the
//!   `cubecl-inspect` binary, and any application that wraps it and resolves
//!   which file to open its own way.
//! - `tui` (feature `tui`) browses the same reports in a terminal, re-read
//!   as the file changes.

mod environment;
mod error;
mod inspector;
mod listing;
mod palette;
mod wordmark;

pub mod command;
pub mod report;
#[cfg(feature = "tui")]
pub mod tui;
pub mod view;

pub use environment::{EnvironmentDirectory, EnvironmentName};
pub use error::InspectError;
pub use inspector::Inspector;
pub use listing::list;
