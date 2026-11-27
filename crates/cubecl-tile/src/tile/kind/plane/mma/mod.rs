//! The manual-mma encoding of a plane tile: the fragment and its transports ([`base`]), its
//! cells written out of registers as whole lines ([`cell_lines`]), the
//! block-scaled instruction's loads and execution ([`block_scaled`]), a fragment deeper than its
//! instruction loaded in lane runs ([`lane_runs`]) or a step at a time ([`steps`]), and the
//! registers' widths ([`widths`]).

pub(crate) mod base;
pub(crate) mod block_scaled;
pub(crate) mod cell_lines;
pub(crate) mod lane_runs;
pub(crate) mod steps;
pub(crate) mod widths;

pub(crate) use base::*;
pub(crate) use block_scaled::*;
pub(crate) use cell_lines::*;
pub(crate) use lane_runs::*;
pub(crate) use steps::*;
