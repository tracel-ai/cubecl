mod branch;
mod cmma;
mod kernel;
mod local_allocator;
mod macros;
mod operation;
mod processing;
mod scope;
mod subcube;
mod synchronization;
mod variable;

pub use branch::*;
pub use cmma::*;
pub use kernel::*;
pub use local_allocator::*;
pub use operation::*;
pub use scope::*;
pub use subcube::*;
pub use synchronization::*;
pub use variable::*;

pub(crate) use macros::cpa;

use std::num::NonZero;
pub type Vectorization = Option<NonZero<u8>>;

impl Item {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Item {
        Item {
            elem: self.elem,
            vectorization,
        }
    }
}
