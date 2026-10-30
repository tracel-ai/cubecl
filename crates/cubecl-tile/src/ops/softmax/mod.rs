//! The softmax reading of a [`Tile`](crate::Tile): [`OnlineSoftmax`] takes a score where its
//! holder keeps it, the ownership read off the score.

pub(crate) mod online;

pub use online::{
    OnlineSoftmax, OnlineSoftmaxExpand, OnlineSoftmaxState, OnlineSoftmaxStateExpand,
};
