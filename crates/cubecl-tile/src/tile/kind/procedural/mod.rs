//! A memory-free tile source evaluated from logical coordinates.

pub(crate) mod affine;
pub(crate) mod base;
pub(crate) mod constant;
pub(crate) mod erased;
pub(crate) mod kind;
pub(crate) mod normalization;
pub(crate) mod phase;
pub(crate) mod product;
pub(crate) mod separable;
pub(crate) mod sum;

pub(crate) use base::*;
pub(crate) use erased::*;
pub(crate) use kind::*;
pub(crate) use normalization::*;
