use alloc::format;
use core::fmt;
use itertools::Itertools;

use cubecl_ir::{
    dialect::OperationExt,
    interfaces::side_effects::{ConditionallySpeculatable, Speculatability},
    prelude::*,
};
use derive_more::{Deref, DerefMut};
use pliron::{
    opts::dce::SideEffects,
    printable::{self, Printable},
};

use crate::{
    BitSetExt, BitsetLatticeExt, SparseBitSet,
    analyses::dataflow_solver::{
        ChangeResult, DataflowSolver, ProgramPoint, ReadRef, WriteRef,
        dense::{DenseBackward, DenseBackwardDataflowAnalysis, DenseLattice, LatticeValue},
        pre::{Computes, op_transitive_kill_set},
    },
};

/// Complement of speculatability, because computing it directly would require a defined universe set
#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct NotSpeculatableSet(pub SparseBitSet);

impl Printable for NotSpeculatableSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "NotSpeculatable({{{}}})", set)
    }
}

impl LatticeValue for NotSpeculatableSet {
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.unite_changed(rhs)
    }
}

pub type SpeculatableAnalysis = DenseBackward<SpeculatableBackward>;
pub type NotSpeculatable = DenseLattice<NotSpeculatableSet>;

#[derive(Default)]
pub struct SpeculatableBackward;

impl DenseBackwardDataflowAnalysis for SpeculatableBackward {
    type LatticeValue = NotSpeculatableSet;

    fn visit_operation(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let before_point = ProgramPoint::before_op(ctx, op);
        let after_point = ProgramPoint::after_op(ctx, op);
        let computes =
            solver.get_or_create_for::<DenseBackward<Self>, Computes>(before_point, after_point);

        if after == before {
            Self::set_to_exit_state(this, solver, ctx, before);
            return Ok(());
        }

        let after = after.deref();
        let after = &after.value().0;

        let has_side_effects = op
            .cast::<dyn SideEffects>(ctx)
            .is_none_or(|op| op.has_side_effects(ctx));
        let is_speculatable = !has_side_effects && is_speculatable(ctx, op);

        let before_set = if is_speculatable {
            after.difference(&computes.deref().value().0).into()
        } else {
            let dependents = op_transitive_kill_set(solver, &computes, op.deref(ctx).results());
            after
                .union(&computes.deref().value().0)
                .union(&dependents)
                .into()
        };
        solver.update_state(ctx, before, |before| {
            before.meet(&NotSpeculatableSet(before_set))
        });
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

fn is_speculatable(ctx: &Context, op: Ptr<Operation>) -> bool {
    let speculatable = match op.cast::<dyn ConditionallySpeculatable>(ctx) {
        Some(op) => op.speculatability(ctx),
        None => Speculatability::NotSpeculatable,
    };
    match speculatable {
        Speculatability::NotSpeculatable => false,
        Speculatability::Speculatable => true,
        Speculatability::RecursivelySpeculatable => is_recursively_speculatable(ctx, op),
    }
}

fn is_recursively_speculatable(ctx: &Context, op: Ptr<Operation>) -> bool {
    for op in op.deref(ctx).immediately_nested_ops(ctx) {
        if !is_speculatable(ctx, op) {
            return false;
        }
    }
    true
}
