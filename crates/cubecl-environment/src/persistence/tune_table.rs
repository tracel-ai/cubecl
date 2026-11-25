//! What names an autotune table.

use super::{Namespace, NamespaceRoot};

/// What the segment numbering an autotune table layout starts with, as in
/// `layout-v1`.
pub const LAYOUT_PREFIX: &str = "layout-v";

/// The layout autotune tables are named by. Bumped when they are named another
/// way.
const LAYOUT: u32 = 1;

/// What names the table one tuner stores its answers in on one hardware.
#[derive(Clone, Copy, Debug)]
pub struct TuneTableName<'a> {
    /// The hardware the answers were measured on, as its `HardwareIdentity`
    /// displays.
    pub hardware: &'a str,
    /// The tuner, named by the code that declared it.
    pub tuner: &'a str,
}

impl TuneTableName<'_> {
    /// `autotune/<version>/layout-v<layout>/<hardware>/<tuner>`.
    pub fn namespace(&self) -> Namespace {
        let Self { hardware, tuner } = self;
        Namespace::scoped(
            NamespaceRoot::AUTOTUNE,
            alloc::format!("{LAYOUT_PREFIX}{LAYOUT}/{hardware}/{tuner}"),
        )
    }
}
