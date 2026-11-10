//! What happens to each vector a merge writes out.

pub(crate) mod base;
pub(crate) mod row_scale;

pub use base::{VectorAt, VectorStage, VectorStageLaunch};
pub use row_scale::RowScale;
