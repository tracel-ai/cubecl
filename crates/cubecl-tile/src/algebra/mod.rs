//! Shared arithmetic: [`Monoid`], [`Semiring`], constant-keeping `Integer` and integer division.

pub(crate) mod constant;
pub(crate) mod division;
pub(crate) mod monoid;
pub(crate) mod semiring;

pub(crate) use constant::Integer;
pub(crate) use constant::{
    IntegerExpand, IntegerSeq, IntegerSeqExpand, constant, fold_add, fold_mul,
};
pub(crate) use division::{floor_div_rem, gcd};
