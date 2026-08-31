use cubecl_ir::prelude::*;
use pliron::{attribute::AttrObj, opts::constants::ConstFoldInterface, printable::Printable};
use smallvec::SmallVec;

use crate::analyses::dataflow_solver::{
    DataflowSolver, ReadRef, WriteRef,
    dead_code::DeadCodeAnalysis,
    sparse::{LatticeValue, SparseForward, SparseForwardDataflowAnalysis, SparseLattice},
};

#[derive(PartialEq, Eq, Clone, Default)]
pub enum ConstantValue {
    #[default]
    Uninitialized,
    Initialized(AttrObj),
    Unknown,
}

impl Printable for ConstantValue {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            ConstantValue::Uninitialized => f.write_str("Uninitialized"),
            ConstantValue::Initialized(attr) => write!(f, "Constant({})", attr.disp(ctx)),
            ConstantValue::Unknown => f.write_str("Unknown"),
        }
    }
}

impl ConstantValue {
    pub fn constant_attr(&self) -> Option<&AttrObj> {
        match self {
            ConstantValue::Unknown | ConstantValue::Uninitialized => None,
            ConstantValue::Initialized(attr) => Some(attr),
        }
    }
}

impl LatticeValue for ConstantValue {
    fn join(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (ConstantValue::Uninitialized, rhs) => rhs.clone(),
            (lhs, ConstantValue::Uninitialized) => lhs.clone(),
            (lhs, rhs) if lhs == rhs => lhs.clone(),
            _ => ConstantValue::Unknown,
        }
    }
}

pub type ConstantLattice = SparseLattice<ConstantValue>;
pub type SparseConstantPropagationAnalysis = SparseForward<SparseConstantPropagation>;

pub struct SparseConstantPropagation;

impl SparseForwardDataflowAnalysis for SparseConstantPropagation {
    type LatticeValue = ConstantValue;

    fn verify(&self, solver: &DataflowSolver, _ctx: &Context, _root: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<DeadCodeAnalysis>()
    }

    fn visit_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        operands: &[ReadRef<ConstantLattice>],
        results: &[WriteRef<ConstantLattice>],
    ) -> Result<()> {
        // This does not mean regions are not supported - `RegionBranchOpInterface` gets handled by
        // the `SparseForward` driver so this is only for ops that can't be reasoned about.
        if op.deref(ctx).num_regions() > 0 {
            this.set_all_to_entry_states(solver, ctx, results);
            return Ok(());
        }

        let dyn_op = op.dyn_op(ctx);
        let Some(fold_op) = op_cast::<dyn ConstFoldInterface>(&*dyn_op) else {
            this.set_all_to_entry_states(solver, ctx, results);
            return Ok(());
        };

        let mut constant_operands =
            SmallVec::<[Option<AttrObj>; 8]>::with_capacity(op.deref(ctx).get_num_operands());
        for operand_lattice in operands {
            match &operand_lattice.deref().value() {
                ConstantValue::Uninitialized => {
                    return Ok(());
                }
                ConstantValue::Unknown => {
                    constant_operands.push(None);
                }
                ConstantValue::Initialized(value) => {
                    constant_operands.push(Some(value.clone()));
                }
            }
        }

        let fold_results = fold_op.check_fold(ctx, &constant_operands);

        for (lattice, result) in results.iter().zip(fold_results) {
            solver.update_state(ctx, lattice, |lattice| match result {
                Some(value) => lattice.join(&ConstantValue::Initialized(value)),
                None => lattice.join(&ConstantValue::Unknown),
            });
        }

        Ok(())
    }

    fn set_to_entry_state(
        _this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<ConstantLattice>,
    ) {
        solver.update_state(ctx, lattice, |it| it.join(&ConstantValue::Unknown));
    }
}
