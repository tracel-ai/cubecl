//! The softmax reading of a [`Tile`](crate::Tile).
//!
//! [`OnlineSoftmax`] folds a score where its holder keeps it, the ownership read off the score.
//! The rest ([`RowState`], [`MaskProbe`], `Tile::softmax`) is the step whose caller states who
//! owns the rows.

pub(crate) mod leaf;
pub(crate) mod online;
pub(crate) mod planewise;
pub(crate) mod rowwise;
pub(crate) mod state;

pub(crate) use state::*;

pub use online::{
    OnlineSoftmax, OnlineSoftmaxExpand, OnlineSoftmaxState, OnlineSoftmaxStateExpand,
    RelayedFactors, RelayedFactorsExpand,
};
pub use state::{LOGIT_MASKED, MaskProbe, RowShare, RowState, masked_recip};
