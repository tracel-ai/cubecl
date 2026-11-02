//! Shared arithmetic: [`Monoid`] and its [`Carrier`]s, [`Semiring`], constant-keeping `Integer` and integer division.

pub(crate) mod carrier;
pub(crate) mod constant;
pub(crate) mod division;
pub(crate) mod indexed;
pub(crate) mod monoid;
pub(crate) mod semiring;
pub(crate) mod top_k;

pub(crate) use constant::Integer;
pub(crate) use constant::{
    IntegerExpand, IntegerSeq, IntegerSeqExpand, constant, simplified_add, simplified_mul,
};
pub(crate) use division::{floor_div_rem, gcd};
