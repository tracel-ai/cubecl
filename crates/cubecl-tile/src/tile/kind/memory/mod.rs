//! The addressable kind ([`Memory`]): gmem and smem tiles, their windows, views and TMA source.

pub(crate) mod access;
pub(crate) mod base;
pub(crate) mod codebook;
pub(crate) mod global;
pub(crate) mod slices;
pub(crate) mod stage;
pub(crate) mod tma;
pub(crate) mod view;
pub(crate) mod window;

pub(crate) use base::*;
pub(crate) use codebook::*;
pub(crate) use global::*;
pub(crate) use stage::*;
pub(crate) use tma::*;
pub(crate) use view::*;
pub(crate) use window::*;
