//! Rewrites the kernel entry ABI to what the JIT host calls.

use cubecl_core::ir::attributes::{ATTR_BUFFER_BINDING, BufferBindingAttr, FuncInterface};
use cubecl_core::ir::dialect::general::BufferLenOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::TypeAttr;
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron_llvm::ops as llvm;
use pliron_llvm::types::PointerType as LlvmPointerType;

use super::prelude::*;
use super::to_llvm::INDEX_WIDTH;

/// `(op, buffer_idx, result)` for each `cube.buffer_len`, gathered during the walk so the ops
/// can be rewritten once the walker no longer holds them borrowed.
#[derive(Default)]
struct BufferLens(Vec<(Ptr<Operation>, usize, Value)>);

/// Collapses buffer args behind `%buffer_ptrs` and lowers `cube.buffer_len` against `%metadata`.
#[derive(Default)]
pub struct LowerEntryAbiPass;

#[pass_name]
impl Pass for LowerEntryAbiPass {
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
        let entry = func.get_entry_block(ctx);

        let num_args = entry.deref(ctx).get_num_arguments();
        let mut buffers: Vec<(usize, usize, Value)> = Vec::new();
        for i in 0..num_args {
            if let Some(binding) =
                func.get_arg_attr::<BufferBindingAttr>(ctx, i, &ATTR_BUFFER_BINDING)
            {
                let pos = binding.buffer_pos;
                buffers.push((i, pos, entry.deref(ctx).get_argument(i)));
            }
        }

        let mut buffer_lens = BufferLens::default();
        visit_all_ops_of_type::<BufferLenOp, _>(ctx, &mut buffer_lens, op, |ctx, state, bl| {
            state
                .0
                .push((bl.get_operation(), bl.buffer_idx(ctx).0, bl.get_result(ctx)));
        });

        let ptr_ty: TypeHandle = LlvmPointerType::get(ctx, 0).into();
        let i64_ty: TypeHandle = IntegerType::get(ctx, INDEX_WIDTH, Signedness::Signless).into();

        let meta_idx = BasicBlock::push_argument(entry, ctx, ptr_ty);
        let meta_ptr = entry.deref(ctx).get_argument(meta_idx);
        for (bl_op, buffer_idx, result) in &buffer_lens.0 {
            let gep = llvm::GetElementPtrOp::new(
                ctx,
                meta_ptr,
                vec![llvm::GepIndex::Constant(*buffer_idx as u32)],
                i64_ty,
            );
            gep.get_operation().insert_before(ctx, *bl_op);
            let load = llvm::LoadOp::new(ctx, gep.get_result(ctx), i64_ty);
            load.get_operation().insert_before(ctx, *bl_op);
            result.replace_all_uses_with(ctx, &load.get_result(ctx));
            Operation::erase(*bl_op, ctx);
        }

        // Collapse buffers behind a single leading `%buffer_ptrs`.
        if !buffers.is_empty() {
            BasicBlock::insert_argument(entry, ctx, 0, ptr_ty);
            let buffer_ptrs = entry.deref(ctx).get_argument(0);
            let terminator = entry
                .deref(ctx)
                .get_terminator(ctx)
                .expect("entry block must be terminated");

            for (_idx, buffer_pos, old_val) in &buffers {
                let gep = llvm::GetElementPtrOp::new(
                    ctx,
                    buffer_ptrs,
                    vec![llvm::GepIndex::Constant(*buffer_pos as u32)],
                    ptr_ty,
                );
                gep.get_operation().insert_before(ctx, terminator);
                let load = llvm::LoadOp::new(ctx, gep.get_result(ctx), ptr_ty);
                load.get_operation().insert_before(ctx, terminator);
                old_val.replace_all_uses_with(ctx, &load.get_result(ctx));
            }

            let mut removed: Vec<usize> = buffers.iter().map(|(i, _, _)| i + 1).collect();
            removed.sort_unstable();
            for idx in removed.into_iter().rev() {
                BasicBlock::remove_argument(entry, ctx, idx);
            }
        }

        let arg_values: Vec<Value> = entry.deref(ctx).arguments().collect();
        let arg_types: Vec<TypeHandle> = arg_values.iter().map(|a| a.get_type(ctx)).collect();
        let res_types = func
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<FunctionType>()
            .expect("FuncOp must have a function type")
            .res_types();
        let new_ty = FunctionType::get(ctx, arg_types, res_types);
        func.set_attr_func_type(ctx, TypeAttr::new(new_ty.into()));

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}
