//! The tile ops in the IR, each with the lowering every backend runs by default.

mod level;
#[cfg(feature = "sequential-levels")]
mod sequential;

pub use level::{LevelOp, LowerLevelPass};
