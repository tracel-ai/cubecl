//! The tile ops in the IR, each lowered by every backend's `LowerOpsPass`.

mod level;
#[cfg(feature = "sequential-levels")]
mod sequential;

pub use level::LevelOp;
