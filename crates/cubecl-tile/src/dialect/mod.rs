//! The tile ops in the IR, each with the lowering every backend runs by default.

mod level;

pub use level::{LevelOp, LowerLevelPass};
pub(crate) use level::TilesExpand;
