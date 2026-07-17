use cubecl_core::ir::dialect::general::BufferLenOp;
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron_llvm::ops as llvm;
use pliron_llvm::types::PointerType as LlvmPointerType;

use super::prelude::*;
use super::to_llvm::INDEX_WIDTH;

/// `(op, buffer_idx, result)` for each `cube.buffer_len`, gathered during the walk so the ops
/// can be rewritten once the walker no longer holds them borrowed.
#[derive(Default)]
struct BufferLens(Vec<(Ptr<Operation>, usize, Value)>);

/// Adds the metadata pointer argument and lowers every `cube.buffer_len` to a load from it.
#[derive(Default)]
pub struct LowerBufferLenPass;

#[pass_name]
impl Pass for LowerBufferLenPass {
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

        let mut buffer_lens = BufferLens::default();
        visit_all_ops_of_type::<BufferLenOp, _>(ctx, &mut buffer_lens, op, |ctx, state, bl| {
            state
                .0
                .push((bl.get_operation(), bl.buffer_idx(ctx).0, bl.get_result(ctx)));
        });
        if buffer_lens.0.is_empty() {
            return Ok(res);
        }

        let ptr_ty = LlvmPointerType::get(ctx, 0).into();
        let arg_id = func.push_argument(ctx, ptr_ty);
        let entry = func.get_entry_block(ctx);
        let meta_ptr = entry.deref(ctx).get_argument(arg_id);

        let i64_ty: TypeHandle = IntegerType::get(ctx, INDEX_WIDTH, Signedness::Signless).into();
        for (bl_op, buffer_idx, result) in buffer_lens.0 {
            let gep = llvm::GetElementPtrOp::new(
                ctx,
                meta_ptr,
                vec![llvm::GepIndex::Constant(buffer_idx as u32)],
                i64_ty,
            );
            gep.get_operation().insert_before(ctx, bl_op);
            let load = llvm::LoadOp::new(ctx, gep.get_result(ctx), i64_ty);
            load.get_operation().insert_before(ctx, bl_op);

            result.replace_all_uses_with(ctx, &load.get_result(ctx));
            Operation::erase(bl_op, ctx);
        }

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}
