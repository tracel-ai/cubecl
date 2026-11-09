//! Packed values in words: the statement ([`base`]) and the unpacking view ([`view`]).

pub(crate) mod base;
pub(crate) mod view;

pub(crate) use base::*;
pub(crate) use view::*;
