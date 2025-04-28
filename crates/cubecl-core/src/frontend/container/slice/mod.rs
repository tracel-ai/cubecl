mod base;
mod operator;

pub use base::*;
pub use operator::*;

pub type SliceMut<E> = Slice<E, ReadWrite>;
