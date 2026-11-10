//! A destination whose vectors pass through a [`Merge`], then through a [`VectorStage`].

pub(crate) mod base;
pub(crate) mod merge;
pub(crate) mod row_values;
pub(crate) mod stage;
pub(crate) mod vectors;

pub use base::{Merged, MergedOperand};
pub use merge::{Merge, MergeLaunch, ReducedRows, RowReduction, RowReductionScratch};
pub use row_values::RowValues;
pub use stage::{RowScale, VectorAt, VectorStage, VectorStageLaunch};
