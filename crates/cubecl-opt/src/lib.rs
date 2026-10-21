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

use cubecl_ir::{
    AddressSpace,
    interfaces::{TypedExt, control_flow::SymbolOpInterface},
    prelude::{IsTerminatorInterface, OperationPtrExt},
};

pub mod analyses;
pub mod passes;
pub mod scoped_map;

use hi_sparse_bitset::{
    Apply, BitSetInterface,
    cache::{DynamicCache, ReduceCache},
    config::Config,
    ops::{self, BitSetOp},
};
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    opts::dce::SideEffects,
    r#type::TypeHandle,
    value::{Use, Value},
};
use smallvec::SmallVec;

pub type SparseBitSet = hi_sparse_bitset::BitSet<hi_sparse_bitset::config::_128bit>;

pub fn reduce_bitset<Conf, Op, I>(op: Op, bitsets: I) -> Option<hi_sparse_bitset::BitSet<Conf>>
where
    Conf: Config,
    Op: BitSetOp,
    I: ExactSizeIterator + Clone,
    I::Item: BitSetInterface<Conf = Conf>,
{
    if bitsets.len() <= <Conf::DefaultCache as ReduceCache>::MAX_LEN {
        hi_sparse_bitset::reduce(op, bitsets).map(Into::into)
    } else {
        hi_sparse_bitset::reduce_w_cache(op, bitsets, DynamicCache).map(Into::into)
    }
}

pub(crate) trait BitSetExt: BitSetInterface {
    #[inline]
    fn difference<Other>(self, other: Other) -> Apply<ops::Sub, Self, Other>
    where
        Other: BitSetInterface<Conf = Self::Conf>,
    {
        hi_sparse_bitset::apply(ops::Sub, self, other)
    }
}
impl<T: BitSetInterface> BitSetExt for T {}

pub(crate) trait BitsetLatticeExt {
    fn set(&mut self, new: Self) -> ChangeResult;
    fn unite_changed(&mut self, other: &Self) -> ChangeResult;
}

impl BitsetLatticeExt for SparseBitSet {
    fn set(&mut self, new: Self) -> ChangeResult
    where
        Self: PartialEq,
    {
        match *self == new {
            true => ChangeResult::Unchanged,
            false => {
                *self = new;
                ChangeResult::Changed
            }
        }
    }

    fn unite_changed(&mut self, other: &Self) -> ChangeResult {
        self.set(self.union(other).into())
    }
}

use crate::analyses::dataflow_solver::ChangeResult;
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

pub fn would_op_be_trivially_dead(ctx: &Context, op: Ptr<Operation>) -> bool {
    if op.impls::<dyn IsTerminatorInterface>(ctx) {
        return false;
    }
    if op.impls::<dyn SymbolOpInterface>(ctx) {
        return false;
    }
    op.cast::<dyn SideEffects>(ctx)
        .is_some_and(|op| !op.has_side_effects(ctx))
}
