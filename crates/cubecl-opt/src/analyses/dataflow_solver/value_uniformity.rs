use core::fmt;

use cubecl_ir::{
    attributes::EntrypointInterface,
    dialect::BlockPtrExt,
    interfaces::{
        control_flow::{CallableOpInterface, RegionBranchOpInterface, RegionSuccessor},
        uniformity::{UniformOpInterface, UniformRegionOpInterface, Uniformity},
    },
    prelude::*,
};
use derive_more::{From, Into};
use pliron::printable::{self, Printable};

use crate::analyses::dataflow_solver::{
    DataflowSolver, ProgramPoint, ReadRef, SmallPtrVec, WriteRef,
    control_flow_uniformity::{BlockUniformity, BlockUniformityAnalysis},
    sparse::{LatticeValue, SparseForward, SparseForwardDataflowAnalysis, SparseLattice},
};

#[derive(PartialEq, Eq, Clone, Copy, Default, From, Into)]
pub struct StrictUniformity(pub Uniformity);

impl Printable for StrictUniformity {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "StrictUniformity::{:?}", self.0)
    }
}

impl LatticeValue for StrictUniformity {
    fn join(&self, rhs: &Self) -> Self {
        Self(self.0.min(rhs.0))
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Default, From, Into)]
pub struct DynamicUniformity(pub Uniformity);

impl Printable for DynamicUniformity {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "DynamicUniformity::{:?}", self.0)
    }
}

impl LatticeValue for DynamicUniformity {
    fn join(&self, rhs: &Self) -> Self {
        Self(self.0.min(rhs.0))
    }
}

pub type DynamicUniformityLattice = SparseLattice<DynamicUniformity>;
pub type DynamicUniformityAnalysis = SparseForward<DynamicValueUniformity>;

pub type StrictUniformityLattice = SparseLattice<StrictUniformity>;
pub type StrictUniformityAnalysis = SparseForward<StrictValueUniformity>;

pub struct DynamicValueUniformity;

impl SparseForwardDataflowAnalysis for DynamicValueUniformity {
    type LatticeValue = DynamicUniformity;

    fn visit_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        operands: &[ReadRef<SparseLattice<Self::LatticeValue>>],
        results: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) -> Result<()> {
        let mut operands_sanitized = SmallPtrVec::with_capacity(operands.len());
        for (operand, lattice) in op.deref(ctx).operands().zip(operands) {
            match operand.defining_block() {
                // We can't properly analyze multi-block regions, so treat all non-entry block args
                // as non-uniform
                Some(block) if !block.is_entry_block(ctx) => {
                    operands_sanitized.push(Uniformity::None);
                }
                _ => operands_sanitized.push(lattice.deref().value().0),
            }
        }

        if let Some(uniform_op) = op_cast::<dyn UniformOpInterface>(&*op.dyn_op(ctx)) {
            let uniformity = uniform_op.uniformity(ctx, &operands_sanitized);
            for result in results {
                solver.update_state(ctx, result, |it| it.join(&uniformity.into()));
            }
        } else {
            this.set_all_to_entry_states(solver, ctx, results);
        }
        Ok(())
    }

    fn visit_callable_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        callable: &dyn CallableOpInterface,
        arg_lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        // All entrypoint args are uniform across the device by default
        if let Some(maybe_entry) = op_cast::<dyn EntrypointInterface>(callable as &dyn Op)
            && maybe_entry.get_entrypoint_abi(ctx).is_some()
        {
            for arg in arg_lattices {
                solver.update_state(ctx, arg, |it| it.join(&Uniformity::Device.into()));
            }
        } else {
            this.visit_callable_operation(solver, ctx, callable, arg_lattices);
        }
    }

    fn visit_region_successors(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        successor: RegionSuccessor,
        lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        this.visit_region_successors(solver, ctx, point, branch, successor, lattices);
        let result_uniformity = match op_cast::<dyn UniformRegionOpInterface>(branch) {
            Some(branch) => {
                let op = branch.get_operation().deref(ctx);
                let operand_lattices = op.operands().map(|operand| {
                    let lattice = this.get_lattice_element(solver, operand);
                    lattice.deref().use_def_subscribe::<SparseForward<Self>>();
                    lattice.deref().value().0
                });
                let operand_lattices: SmallPtrVec<_> = operand_lattices.collect();
                branch.result_uniformity(ctx, &operand_lattices)
            }
            None => Uniformity::None,
        };
        for lattice in lattices {
            solver.update_state(ctx, lattice, |it| it.join(&result_uniformity.into()));
        }
    }

    fn set_to_entry_state(
        _this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<SparseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.join(&Uniformity::None.into()));
    }
}

pub struct StrictValueUniformity;

impl SparseForwardDataflowAnalysis for StrictValueUniformity {
    type LatticeValue = StrictUniformity;

    fn verify(&self, solver: &DataflowSolver, _ctx: &Context, _root: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<BlockUniformityAnalysis>()?;
        solver.require_loaded::<SparseForward<DynamicValueUniformity>>()?;
        Ok(())
    }

    // Strict uniformity = dynamic uniformity ∩ parent block uniformity
    fn visit_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        _operands: &[ReadRef<SparseLattice<Self::LatticeValue>>],
        results: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) -> Result<()> {
        let Some(block) = op.deref(ctx).get_parent_block() else {
            return Ok(());
        };

        if op.deref(ctx).num_regions() > 0 {
            this.set_all_to_entry_states(solver, ctx, results);
            return Ok(());
        }

        let block_lattice = solver.get_or_create_for::<SparseForward<Self>, BlockUniformity>(
            ProgramPoint::after_op(ctx, op),
            block,
        );
        let block_uniformity = block_lattice.deref().value();

        for (result, lattice) in op.deref(ctx).operands().zip(results) {
            let dynamic = solver.get_or_create::<DynamicUniformityLattice>(result);
            dynamic.deref().use_def_subscribe::<SparseForward<Self>>();
            let uniformity = dynamic.deref().value().0.min(block_uniformity);
            solver.update_state(ctx, lattice, |it| it.join(&uniformity.into()));
        }
        Ok(())
    }

    fn visit_callable_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        callable: &dyn CallableOpInterface,
        arg_lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        // All entrypoint args are uniform across the device by default, but for strict uniformity
        // the largest meaningful scope is `Cube`
        if let Some(maybe_entry) = op_cast::<dyn EntrypointInterface>(callable as &dyn Op)
            && maybe_entry.get_entrypoint_abi(ctx).is_some()
        {
            for arg in arg_lattices {
                solver.update_state(ctx, arg, |it| it.join(&Uniformity::Cube.into()));
            }
        } else {
            this.visit_callable_operation(solver, ctx, callable, arg_lattices);
        }
    }

    fn visit_region_successors(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        successor: RegionSuccessor,
        lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        this.visit_region_successors(solver, ctx, point, branch, successor, lattices);
        let result_uniformity = match op_cast::<dyn UniformRegionOpInterface>(branch) {
            Some(branch) => {
                let op = branch.get_operation().deref(ctx);
                let operand_lattices = op.operands().map(|operand| {
                    let lattice = this.get_lattice_element(solver, operand);
                    lattice.deref().use_def_subscribe::<SparseForward<Self>>();
                    lattice.deref().value().0
                });
                let operand_lattices: SmallPtrVec<_> = operand_lattices.collect();
                branch.result_uniformity(ctx, &operand_lattices)
            }
            None => Uniformity::None,
        };
        for lattice in lattices {
            solver.update_state(ctx, lattice, |it| it.join(&result_uniformity.into()));
        }
    }

    fn set_to_entry_state(
        _this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<SparseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.join(&Uniformity::None.into()));
    }
}
