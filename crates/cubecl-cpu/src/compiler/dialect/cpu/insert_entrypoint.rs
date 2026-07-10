use core::any::type_name;

use cubecl_core::ir::attributes::EntrypointInterface;
use cubecl_core::ir::dialect::branch::YieldOp;
use cubecl_core::ir::prelude::*;
use pliron::basic_block::BasicBlock;
use pliron::builtin::ops::FuncOp;
use pliron::linked_list::ContainsLinkedList;

use crate::compiler::dialect::cpu::entrypoint::EntrypointOp;

/// Wraps the body of the kernel entry function in a [`EntrypointOp`] (`cpu.entrypoint`).
///
/// On CPU there are no hardware threads, so a single invocation must eventually loop over every
/// cube position and unit and run the kernel body per unit. This pass performs the first half of
/// that transformation: it inserts a `cpu.entrypoint` op around the existing body so a later
/// lowering pass can expand it into the grid-iteration loop nest (see `cpu_entry_point`).
///
/// After this pass the entry function looks like:
///
/// ```text
/// func @kernel(args...) {
/// entry:
///   cpu.entrypoint {
///   ^body:
///     <all original body ops>
///     branch.yield
///   }
///   return
/// }
/// ```
#[derive(Default)]
pub struct InsertEntrypointPass;

impl Pass for InsertEntrypointPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let Some(func) = op.as_op::<FuncOp>(ctx) else {
            return Ok(res);
        };
        if func.get_entrypoint_abi(ctx).is_none() {
            return Ok(res);
        }

        let entry_block = func.get_entry_block(ctx);
        let terminator = entry_block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("Entry block should be terminated with a return");

        let body_ops: Vec<Ptr<Operation>> = entry_block
            .deref(ctx)
            .iter(ctx)
            .filter(|o| *o != terminator)
            .collect();

        let entrypoint = EntrypointOp::new(ctx);
        let body_block: Ptr<BasicBlock> = entrypoint.body(ctx);

        for body_op in body_ops {
            body_op.unlink(ctx);
            body_op.insert_at_back(body_block, ctx);
        }
        YieldOp::new(ctx)
            .get_operation()
            .insert_at_back(body_block, ctx);

        entrypoint.get_operation().insert_before(ctx, terminator);

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}
