//! The reports in a terminal browser: the environments to choose from first,
//! then one tab per report, the autotune keys selectable, and everything read
//! again whenever the file changes — open it beside a running session and
//! watch it fill the file.

mod base;
mod picker;
mod theme;
mod viewer;

pub use base::*;
