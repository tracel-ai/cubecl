use cubecl_ir::{dialect::BlockPtrExt, prelude::*};
use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::attr_interfaces::MaterializableAttr,
    irbuild::match_rewrite::apply_match_rewrite,
    linked_list::ContainsLinkedList,
    opts::constants::ConstFoldInterface,
    region::Region,
};
use smallvec::SmallVec;

use crate::analyses::dataflow_solver::{
    DataflowSolver, SmallPtrVec, SolverConfig,
    dead_code::DeadCodeAnalysis,
    sccp::{ConstantLattice, SparseConstantPropagationAnalysis},
};

fn const_value(solver: &DataflowSolver, value: Value) -> Option<AttrObj> {
    let lattice = solver.lookup_state::<ConstantLattice>(value)?;
    lattice.deref().value().constant_attr().cloned()
}

fn const_operands(
    solver: &DataflowSolver,
    ctx: &Context,
    op: Ptr<Operation>,
) -> SmallPtrVec<Option<AttrObj>> {
    let mut out = SmallVec::with_capacity(op.deref(ctx).get_num_operands());
    for operand in op.deref(ctx).operands() {
        out.push(const_value(solver, operand));
    }
    out
}

fn const_results(
    solver: &DataflowSolver,
    ctx: &Context,
    op: Ptr<Operation>,
) -> SmallVec<[(Value, AttrObj); 4]> {
    let mut out = SmallVec::with_capacity(op.deref(ctx).get_num_operands());
    for result in op.deref(ctx).results() {
        let Some(value) = const_value(solver, result) else {
            continue;
        };
        out.push((result, value));
    }
    out
}

fn rewrite(
    solver: &DataflowSolver,
    ctx: &mut Context,
    initial_regions: &[Ptr<Region>],
) -> Result<IRStatus> {
    let mut status = IRStatus::Unchanged;
    let mut rewriter = IRRewriter::<Recorder>::default();

    let mut worklist = Vec::new();
    let add_to_worklist = |ctx: &Context, worklist: &mut Vec<_>, regions: &[Ptr<Region>]| {
        for region in regions {
            worklist.extend(region.deref(ctx).iter(ctx).rev());
        }
    };

    add_to_worklist(ctx, &mut worklist, initial_regions);
    while let Some(block) = worklist.pop() {
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for op in ops.into_iter().rev() {
            if let Some(const_fold) = op_cast::<dyn ConstFoldInterface>(&*op.dyn_op(ctx)) {
                rewriter.set_insertion_point_before_operation(op);
                let operands = const_operands(solver, ctx, op);
                status |= const_fold.fold_in_place(ctx, &operands, &mut rewriter);
            } else {
                rewriter.set_insertion_point_after_operation(op);
                let results = const_results(solver, ctx, op);
                for (result, value) in results {
                    if let Some(materializable) = attr_cast::<dyn MaterializableAttr>(&*value) {
                        let op = materializable.materialize(ctx);
                        rewriter.append_operation(ctx, op);
                        rewriter.replace_value_uses_with(ctx, result, op.result(ctx));
                        status |= IRStatus::Changed;
                    }
                }
            }

            add_to_worklist(ctx, &mut worklist, &op.regions(ctx));
        }

        rewriter.set_insertion_point_to_block_start(block);
        for arg in block.arguments(ctx) {
            let Some(const_val) = const_value(solver, arg) else {
                continue;
            };
            let Some(materializable) = attr_cast::<dyn MaterializableAttr>(&*const_val) else {
                continue;
            };
            let op = materializable.materialize(ctx);
            rewriter.append_operation(ctx, op);
            rewriter.replace_value_uses_with(ctx, arg, op.result(ctx));
            status |= IRStatus::Changed;
        }
    }

    Ok(status)
}

pub fn sccp(root_op: Ptr<Operation>, ctx: &mut Context) -> Result<IRStatus> {
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DeadCodeAnalysis::default());
    solver.load(SparseConstantPropagationAnalysis::default());
    solver.initialize_and_run(ctx, root_op)?;
    rewrite(&solver, ctx, &root_op.regions(ctx))
}

pub struct SCCPPass;

#[pass_name]
impl Pass for SCCPPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        res.ir_changed |= sccp(op, ctx)?;
        res.ir_changed |= apply_match_rewrite(ctx, &mut Canonicalize, Default::default(), op)?;
        Ok(res)
    }
}
