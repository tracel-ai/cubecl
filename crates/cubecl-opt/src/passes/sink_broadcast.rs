//! Runs a pure lane-wise op once on the scalar when every vector it reads is a broadcast.
//!
//! `op(broadcast(a), broadcast(b))` becomes `broadcast(op(a, b))`. Every lane of the vector op
//! computes `op(a, b)`, which is what the unroll pass relies on to split it lane by lane, so the
//! rewrite gives the same bits. It lets a vector kernel fold what a one-lane kernel folds: an op
//! on constants alone, which constant propagation folds only once it is scalar, and a constant
//! operand the identities in `SimplifyOps` see. Without it, a kernel generic over its vector size
//! reaches the multiply-add fusion with different operations at one lane and at more, and fuses
//! different products.

use alloc::vec::Vec;
use cubecl_ir::{
    NamedRewrite,
    dialect::vector::VectorBroadcastOp,
    interfaces::{Deterministic, Synchronizes, TriviallyUnrollable, side_effects::MemoryEffectsOp},
    prelude::{Inserter as _, Rewriter as _, *},
    rewrite::MatchRewritePass,
    try_cast_op,
    types::VectorType,
};
use pliron::opts::dce::SideEffects;

/// Sinks vector broadcasts through the pure lane-wise ops that read them.
pub type SinkBroadcastPass = MatchRewritePass<SinkBroadcast>;

#[derive(Default, Clone, Copy, NamedRewrite)]
pub struct SinkBroadcast;

/// The scalar `value` is a broadcast of, if it is one.
fn broadcast_input(ctx: &Context, value: Value) -> Option<Value> {
    let broadcast = value.defining_op()?.as_op::<VectorBroadcastOp>(ctx)?;
    Some(broadcast.input(ctx))
}

fn is_vector(ctx: &Context, value: Value) -> bool {
    value.get_type(ctx).deref(ctx).is::<VectorType>()
}

/// Whether `op` computes each lane from the same lane of its operands and nothing else: no
/// memory, no side effect, no other unit of the plane.
fn is_pure_lane_wise(ctx: &Context, op: Ptr<Operation>) -> bool {
    let operation = op.deref(ctx);
    if operation.num_regions() > 0 || operation.get_num_results() != 1 {
        return false;
    }
    drop(operation);
    if op.as_op::<VectorBroadcastOp>(ctx).is_some() {
        return false;
    }
    let dyn_op = op.dyn_op(ctx);
    op_cast::<dyn TriviallyUnrollable>(&*dyn_op).is_some()
        && op_cast::<dyn Deterministic>(&*dyn_op).is_some()
        && op_cast::<dyn Synchronizes>(&*dyn_op).is_none()
        && op_cast::<dyn SideEffects>(&*dyn_op).is_some_and(|it| !it.has_side_effects(ctx))
        && op_cast::<dyn MemoryEffectsOp>(&*dyn_op).is_some_and(|it| !it.has_effects(ctx))
}

/// The operands and result type of `op` run on one lane, when it reads at least one vector and
/// every vector it reads is a broadcast.
fn scalar_form(ctx: &Context, op: Ptr<Operation>) -> Option<(Vec<Value>, TypeHandle)> {
    if !is_pure_lane_wise(ctx, op) {
        return None;
    }
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let scalar_ty = result_ty.deref(ctx).downcast_ref::<VectorType>()?.inner;

    let operands = op.deref(ctx).operands().collect::<Vec<_>>();
    let mut reads_vector = false;
    let mut scalars = Vec::with_capacity(operands.len());
    for operand in operands {
        if is_vector(ctx, operand) {
            reads_vector = true;
            scalars.push(broadcast_input(ctx, operand)?);
        } else {
            scalars.push(operand);
        }
    }
    reads_vector.then_some((scalars, scalar_ty))
}

impl MatchRewrite for SinkBroadcast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        scalar_form(ctx, op).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some((operands, scalar_ty)) = scalar_form(ctx, op) else {
            return Ok(());
        };
        let dyn_op = op.dyn_op(ctx);
        let rematerialize = try_cast_op!(dyn_op, ctx, dyn TriviallyUnrollable);
        let attributes = op.deref(ctx).attributes.clone();
        let scalar = rematerialize.materialize(ctx, alloc::vec![scalar_ty], operands, attributes);
        rewriter.insert_operation(ctx, scalar);

        let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
        let lane = scalar.deref(ctx).get_result(0);
        let broadcast = VectorBroadcastOp::new(ctx, result_ty, lane);
        rewriter.insert_op(ctx, &broadcast);
        rewriter.replace_operation(ctx, op, broadcast.get_operation());
        Ok(())
    }
}
