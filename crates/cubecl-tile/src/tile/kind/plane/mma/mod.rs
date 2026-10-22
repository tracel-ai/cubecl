//! The manual-mma encoding of a plane tile: the fragment and its transports ([`base`]), the
//! block-scaled instruction's loads and execution ([`block_scaled`]), a fragment deeper than its
//! instruction loaded in lane runs ([`lane_runs`]), and the registers' widths ([`widths`]).

pub(crate) mod base;
pub(crate) mod block_scaled;
pub(crate) mod lane_runs;
pub(crate) mod steps;
mod widths;

pub(crate) use base::*;
pub(crate) use block_scaled::*;
pub(crate) use lane_runs::*;
pub(crate) use steps::*;
