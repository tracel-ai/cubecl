use cubecl_ir::prelude::*;
use pliron::{attribute::AttrObj, opts::constants::ConstFoldInterface, printable::Printable};
use smallvec::SmallVec;

use crate::analyses::dataflow_solver::{
    DataflowAnalysis, DataflowSolver, ReadRef, WriteRef,
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
            ConstantValue::Uninitialized => f.write_str("ConstantValue::Uninitialized"),
            ConstantValue::Initialized(attr) => {
                write!(f, "ConstantValue::Constant({})", attr.disp(ctx))
            }
            ConstantValue::Unknown => f.write_str("ConstantValue::Unknown"),
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
    fn join(this: &SparseLattice<Self>, rhs: &Self) -> Self {
        match (this.value(), rhs) {
            (ConstantValue::Uninitialized, rhs) => rhs.clone(),
            (lhs, ConstantValue::Uninitialized) => lhs.clone(),
            (lhs, rhs) if lhs == rhs => lhs.clone(),
            _ => ConstantValue::Unknown,
        }
    }
}

pub type ConstantLattice = SparseLattice<ConstantValue>;
pub type SparseConstantPropagationAnalysis = SparseForward<SparseConstantPropagation>;

#[derive(Default)]
pub struct SparseConstantPropagation;

impl SparseForwardDataflowAnalysis for SparseConstantPropagation {
    type LatticeValue = ConstantValue;

    fn on_load(&self, solver: &mut DataflowSolver) {
        solver.ensure_loaded(DeadCodeAnalysis::default);
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

/// A dummy version of SCCP that can be used by [`DeadCode`] when constant propagation is not required.
/// Every analysis depends on dead code, so this can make it cheaper when dead branches are
/// irrelevant, i.e. only the [`PredecessorState`] is needed, or the analysis is run after constant
/// branches have already been folded.
///
/// Should be loaded before [`DeadCode`] so that can immediately propagate into all branches.
#[derive(Default)]
pub struct DummySparseConstantPropagation;

impl DataflowAnalysis for DummySparseConstantPropagation {
    fn initialize(
        &mut self,
        mut solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        visit_all_values(ctx, &mut solver, root, |ctx, solver, value| {
            let lattice = solver.get_or_create_mut::<ConstantLattice>(value);
            solver.update_state(ctx, &lattice, |it| it.join(&ConstantValue::Unknown));
        });
        Ok(())
    }

    fn visit(&self, _: &DataflowSolver, _: &Context, _: super::ProgramPoint) -> Result<()> {
        Ok(())
    }
}
