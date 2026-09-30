use alloc::vec::Vec;
use hashbrown::HashSet;

use cubecl_ir::{
    dialect::scf::ForOp,
    interfaces::{MemoryEffects, Synchronizes},
    prelude::*,
};
use pliron::{
    basic_block::BasicBlock,
    graph::walkers::{IRNode, WALKCONFIG_POSTORDER_FORWARD, uninterruptible::immutable::walk_op},
    linked_list::ContainsLinkedList,
    opts::dce::SideEffects,
    region::Region,
};

/// Loop-Invariant Code Motion (LICM) optimization pass.
///
/// Detects structured loops (`scf::ForOp`), identifies pure operations whose inputs do not
/// depend on loop iterations, and hoists them immediately before the loop in topological order.
#[derive(Default, Clone, Copy, Debug)]
pub struct LoopInvariantCodeMotionPass;

#[pass_name]
impl Pass for LoopInvariantCodeMotionPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut any_ir_changed = IRStatus::Unchanged;

        // Traverse loops in post-order forward so that innermost loops are processed first,
        // allowing hoisted operations to bubble outward through nested loops.
        let mut loops = Vec::new();
        walk_op(
            ctx,
            &mut loops,
            &WALKCONFIG_POSTORDER_FORWARD,
            op,
            |ctx, loops, node| {
                if let IRNode::Operation(op_ptr) = node
                    && op_ptr.is_op::<ForOp>(ctx)
                {
                    loops.push(op_ptr);
                }
            },
        );

        for loop_op in loops {
            let Some(for_op) = loop_op.as_op::<ForOp>(ctx) else {
                continue;
            };
            let body_block = for_op.loop_body(ctx);
            let loop_region = for_op.loop_region(ctx);

            // Collect block arguments of the entry block (IV and loop-carried values).
            // These are inherently variant.
            let body_args: HashSet<Value> = body_block.deref(ctx).arguments().collect();
            let mut hoisted_values = HashSet::new();

            // Fixpoint worklist loop over the entry block's operations.
            let mut changed = true;
            while changed {
                changed = false;
                let ops: Vec<Ptr<Operation>> = body_block.deref(ctx).iter(ctx).collect();

                for cur_op in ops {
                    // 1. Skip terminators (YieldOp, etc.).
                    if cur_op.is_terminator(ctx) {
                        continue;
                    }

                    // 2. Purity check.
                    if !is_pure(ctx, cur_op) {
                        continue;
                    }

                    // 3. Invariance check on operands.
                    if are_operands_invariant(
                        ctx,
                        cur_op,
                        loop_op,
                        loop_region,
                        body_block,
                        &body_args,
                        &hoisted_values,
                    ) {
                        // 4. Hoist operation immediately before the loop.
                        cur_op.unlink(ctx);
                        cur_op.insert_before(ctx, loop_op);

                        for res in cur_op.results(ctx) {
                            hoisted_values.insert(res);
                        }

                        changed = true;
                        any_ir_changed = IRStatus::Changed;
                    }
                }
            }
        }

        let mut res = PassResult::default();
        res.ir_changed = any_ir_changed;
        Ok(res)
    }
}

/// Checks whether an operation is pure and safe to hoist.
///
/// Disallows any operation that writes memory, synchronizes, or contains nested regions.
fn is_pure(ctx: &Context, op: Ptr<Operation>) -> bool {
    // Terminators cannot be hoisted.
    if op.is_terminator(ctx) {
        return false;
    }

    // Operations with nested regions are disallowed.
    if op.deref(ctx).num_regions() > 0 {
        return false;
    }

    // Operations that synchronize are disallowed.
    if op.impls::<dyn Synchronizes>(ctx) {
        return false;
    }

    let dyn_op = op.dyn_op(ctx);

    // Must implement SideEffects interface and has_side_effects must be false.
    let Some(side_effects) = op_cast::<dyn SideEffects>(&*dyn_op) else {
        return false;
    };
    if side_effects.has_side_effects(ctx) {
        return false;
    }

    // Must implement MemoryEffects interface and have no memory effects.
    let Some(memory_effects) = op_cast::<dyn MemoryEffects>(&*dyn_op) else {
        return false;
    };
    if memory_effects.has_effects(ctx) {
        return false;
    }

    true
}

/// Checks whether all operands of `op` are invariant with respect to the loop.
fn are_operands_invariant(
    ctx: &Context,
    op: Ptr<Operation>,
    loop_op: Ptr<Operation>,
    loop_region: Ptr<Region>,
    body_block: Ptr<BasicBlock>,
    body_args: &HashSet<Value>,
    hoisted_values: &HashSet<Value>,
) -> bool {
    for operand in op.operands(ctx) {
        if !is_operand_invariant(
            ctx,
            operand,
            loop_op,
            loop_region,
            body_block,
            body_args,
            hoisted_values,
        ) {
            return false;
        }
    }
    true
}

/// Determines if a single `operand` is loop-invariant.
fn is_operand_invariant(
    ctx: &Context,
    operand: Value,
    loop_op: Ptr<Operation>,
    loop_region: Ptr<Region>,
    body_block: Ptr<BasicBlock>,
    body_args: &HashSet<Value>,
    hoisted_values: &HashSet<Value>,
) -> bool {
    // Already hoisted in the current fixpoint pass.
    if hoisted_values.contains(&operand) {
        return true;
    }

    // Loop entry block arguments (induction variable and loop-carried values) are inherently variant.
    if body_args.contains(&operand) {
        return false;
    }

    // Check defining operation, if any.
    if let Some(def_op) = operand.defining_op() {
        if def_op == loop_op {
            return false;
        }
        // Still inside loop body block (not hoisted).
        if def_op.deref(ctx).get_parent_block() == Some(body_block) {
            return false;
        }
        // Inside the loop's hierarchy.
        if loop_op.is_ancestor_of(ctx, def_op) {
            return false;
        }
        // Defined outside the loop.
        return true;
    }

    // Check defining block for block arguments (e.g. function arguments).
    if let Some(def_block) = operand.get_defining_block(ctx) {
        if def_block == body_block || def_block.deref(ctx).get_parent_region() == Some(loop_region)
        {
            return false;
        }
        if let Some(parent_op) = def_block.deref(ctx).get_parent_op(ctx)
            && (parent_op == loop_op || loop_op.is_ancestor_of(ctx, parent_op))
        {
            return false;
        }
        // Defined in a block outside the loop.
        return true;
    }

    // Conservatively assume variant if definition cannot be resolved.
    false
}
