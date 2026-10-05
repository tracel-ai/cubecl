//! Analysis to track dependent value number sets for each value, so kill sets can be efficiently computed

use core::fmt;

use alloc::format;
use cubecl_ir::prelude::{Operation, Result, Use, Value, *};
use derive_more::{Deref, DerefMut};
use itertools::Itertools;
use pliron::printable::{self, Printable};

use crate::{
    SparseBitSet,
    analyses::dataflow_solver::{
        ProgramPoint,
        sparse::{LatticeValue, SparseBackward, SparseBackwardDataflowAnalysis, SparseLattice},
        value_numbering::ValueNumberLattice,
    },
};

use super::{DataflowSolver, ReadRef, WriteRef};

#[derive(Deref, DerefMut, PartialEq, Default, Clone)]
pub struct Dependents(pub SparseBitSet);

impl Printable for Dependents {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Dependents({{{}}})", set)
    }
}

impl LatticeValue for Dependents {
    fn join(this: &SparseLattice<Self>, rhs: &Self) -> Self {
        Dependents(this.value().intersection(&rhs.0).into())
    }

    fn meet(this: &SparseLattice<Self>, rhs: &Self) -> Option<Self> {
        Some(Dependents(this.value().union(&rhs.0).into()))
    }
}

pub type DependentsLattice = SparseLattice<Dependents>;
pub type ValueDependentsAnalysis = SparseBackward<ValueDependents>;

#[derive(Default)]
pub struct ValueDependents;

impl SparseBackwardDataflowAnalysis for ValueDependents {
    type LatticeValue = Dependents;

    fn visit_operation(
        _: &SparseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        results: &[ReadRef<SparseLattice<Self::LatticeValue>>],
        operands: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) -> Result<()> {
        let point = ProgramPoint::after_op(ctx, op);

        for result_lattice in results {
            let result = result_lattice.deref().anchor();
            let value =
                solver.get_or_create_for::<SparseBackward<Self>, ValueNumberLattice>(point, result);
            let value = match value.deref().value().value() {
                Some(value) => value,
                None => return Ok(()),
            };
            let mut res_set = result_lattice.deref().value().clone();
            res_set.insert(value as usize);
            for operand in operands {
                solver.update_state(ctx, operand, |it| it.meet(&res_set));
            }
        }

        Ok(())
    }

    fn visit_branch_operand(
        this: &SparseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        operand: Use<Value>,
    ) {
        let lattice = this.get_lattice_element_mut(solver, operand.get_def(ctx));
        Self::set_to_exit_state(this, solver, ctx, &lattice);
    }

    fn visit_call_operand(
        this: &SparseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        operand: Use<Value>,
    ) {
        let lattice = this.get_lattice_element_mut(solver, operand.get_def(ctx));
        Self::set_to_exit_state(this, solver, ctx, &lattice);
    }

    fn set_to_exit_state(
        _this: &SparseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<SparseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.meet(&Default::default()));
    }
}
