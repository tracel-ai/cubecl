mod accum;
mod collect;
mod indexed;
mod iterator;

pub(crate) use self::indexed::expand_indexed;
pub use self::{
    accum::IntoNumericExpand,
    collect::IntoIteratorExpand,
    indexed::IndexedIteratorExpand,
    iterator::{IterArg, IteratorExpand},
};
