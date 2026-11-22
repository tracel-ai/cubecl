//! Analysis to track dependent value number sets for each value, so kill sets can be efficiently computed

use core::fmt;

use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    AddressSpace,
    interfaces::aliasing::PointerExt,
    prelude::{Operation, Result, Value, *},
    types::PointerType,
};
use derive_more::{Deref, DerefMut, From};
use pliron::{
    graph::walkers::uninterruptible::immutable::walk_op,
    opts::mem2reg::PromotableAllocationInterface,
    printable::{self, Printable},
};

use crate::{BitsetLatticeExt, SparseBitSet, analyses::dataflow_solver::dense::*};

use super::{DataflowSolver, ReadRef, WriteRef};

#[derive(Deref, DerefMut, PartialEq, Default, Clone, From)]
pub struct PointerLivenessSet(pub SparseBitSet);

impl Printable for PointerLivenessSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Liveness({{{:?}}})", self.0)
    }
}

impl LatticeValue for PointerLivenessSet {
    fn meet(&mut self, rhs: &Self) -> super::ChangeResult {
        self.0.unite_changed(&rhs.0)
    }
}

pub type LivenessLattice = DenseLattice<PointerLivenessSet>;
pub type PointerLivenessAnalysis = DenseBackward<PointerLiveness>;

type HMap<K, V> = HashMap<K, V>;

#[derive(Default)]
pub struct PointerLiveness {
    all: SparseBitSet,
    pub roots: HMap<Value, usize>,
}

impl DenseBackwardDataflowAnalysis for PointerLiveness {
    type LatticeValue = PointerLivenessSet;

    fn initialize(
        this: &mut DenseBackward<Self>,
        _: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) {
        let mut idx = 0usize;
        let mut all = SparseBitSet::new();
        let mut roots = HMap::new();

        walk_op(
            ctx,
            &mut (&mut idx, &mut all, &mut roots),
            &WALKCONFIG_PREORDER_FORWARD,
            root,
            |ctx, (idx, all, roots), node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                let Some(promotable) = op.cast::<dyn PromotableAllocationInterface>(ctx) else {
                    return;
                };
                let ptrs = promotable.alloc_info(ctx).into_iter().map(|it| it.ptr);
                for ptr in ptrs {
                    roots.insert(ptr, **idx);
                    all.insert(**idx);
                    **idx += 1;
                }
            },
        );

        this.all = all;
        this.roots = roots;
    }

    fn visit_operation(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let op_ = op.deref(ctx);
        let uses = op_.operands().filter_map(|opd| {
            let ty = opd.get_type(ctx).deref(ctx);
            let ty = ty.downcast_ref::<PointerType>()?;
            match ty.address_space {
                AddressSpace::Local => Some(opd.get_root_ptr(ctx)),
                _ => None,
            }
        });
        let mut used_roots = after.deref().value().clone();
        for ptr in uses {
            match this.roots.get(&ptr) {
                Some(&root_idx) => used_roots.insert(root_idx),
                None => used_roots.unite(&this.all),
            }
        }
        solver.update_state(ctx, before, |it| it.meet(&used_roots));
        Ok(())
    }

    fn set_to_exit_state(
        _: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.meet(&Default::default()));
    }
}
