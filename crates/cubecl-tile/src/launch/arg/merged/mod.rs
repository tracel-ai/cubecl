//! A destination whose lines pass through a [`Merge`], then through a [`LineStage`].

pub(crate) mod base;
pub(crate) mod lines;
pub(crate) mod merge;
pub(crate) mod stage;

pub use base::{Merged, MergedOperand};
pub use merge::{Merge, MergeLaunch};
pub use stage::{LineAt, LineStage, LineStageLaunch};
