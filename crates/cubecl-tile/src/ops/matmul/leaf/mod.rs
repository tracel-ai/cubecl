//! Plane-level contraction leaves: hardware fragments ([`cmma`], [`mma`]) and the register nest
//! ([`registers`]) with its [`promoted`] and [`memory`] accumulator forms.

pub(crate) mod cmma;
pub(crate) mod memory;
pub(crate) mod mma;
pub(crate) mod promoted;
pub(crate) mod registers;
pub(crate) mod scale;

pub(crate) use cmma::{FragmentRead, rhs_layout};
pub(crate) use mma::{deep_load, window_layouts};
pub(crate) use scale::Side;
pub(crate) use scale::{check_scales_omit_rather_than_divide, check_scales_ride};
