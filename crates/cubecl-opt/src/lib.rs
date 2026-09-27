//! # `CubeCL` Optimizer
//!
//! Contains custom optimization passes.

#![no_std]
#![allow(
    clippy::result_unit_err,
    reason = "semantically useful for optimization passes"
)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

use core::ops::Range;

use cubecl_ir::{AddressSpace, interfaces::TypedExt};

pub mod analyses;
pub mod passes;
pub mod scoped_map;

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    r#type::TypeHandle,
    value::{Use, Value},
};
use smallvec::SmallVec;

// pub(crate) type SparseBitSet = hi_sparse_bitset::BitSet<hi_sparse_bitset::config::_128bit>;

pub use crate::analyses::liveness::shared::SharedLiveness;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryResource {
    pub address_space: AddressSpace,
    pub value_ty: TypeHandle,
    pub alignment: usize,
    /// The root pointer value returned from the allocation or passed into the kernel. All other
    /// pointers into the same value are derived from this.
    pub root_ptr: Value,
}

impl MemoryResource {
    /// The byte size of this shared memory
    pub fn size(&self, ctx: &Context) -> usize {
        self.value_ty.size(ctx)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferVisibility {
    /// Whether the buffer is ever read from
    pub readable: bool,
    /// Whether the buffer is ever written to
    pub writable: bool,
}

pub(crate) fn operand_range_to_uses(
    ctx: &Context,
    op: Ptr<Operation>,
    range: Range<usize>,
) -> SmallVec<[Use<Value>; 4]> {
    let op = op.deref(ctx);
    range.map(|idx| op.get_operand_as_use(idx)).collect()
}
