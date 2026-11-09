//! How a fill and the read it feeds meet: who waits for whom, and on what.

pub(crate) mod base;
#[cfg(test)]
pub(crate) mod protocol;

pub(crate) use base::*;
