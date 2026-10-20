//! Reports rendered as text: `Text(&report)` implements `Display` for every
//! report the [command line](crate::command) prints.
//!
//! The alternate form, `{:#}`, paints the text for a terminal, and the flag
//! travels only through the formatter: a rendering nested in another is
//! written with `fmt::Display::fmt(&nested, f)`, which hands it `f` and its
//! flags. `write!(f, "{nested}")` starts a formatter of its own, and what it
//! writes comes out unpainted inside a painted report.

mod autotune;
mod base;
mod diff;
mod fields;
mod kernels;
mod key;
mod memory;
mod paint;
mod summary;
mod table;
mod timeline;

pub use base::*;
pub use key::KeyText;
#[cfg(all(test, feature = "tui"))]
pub(crate) use paint::Role;
#[cfg(feature = "tui")]
pub(crate) use table::Table;
