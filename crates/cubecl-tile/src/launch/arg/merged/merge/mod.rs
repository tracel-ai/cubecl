//! How the vectors a routine stores become the output.

pub(crate) mod base;
pub(crate) mod box_holder;
pub(crate) mod reduced_rows;
pub(crate) mod row_reduction;
pub(crate) mod row_reduction_scratch;

pub use base::{Merge, MergeLaunch};
pub use reduced_rows::ReducedRows;
pub use row_reduction::RowReduction;
pub use row_reduction_scratch::RowReductionScratch;
