//! Rewrites the kernel entry ABI to what the JIT host calls.

use core::cell::RefCell;
use cubecl_core::ir::attributes::{ATTR_BUFFER_BINDING, BufferBindingAttr, FuncInterface};
use cubecl_core::ir::dialect::general::{BufferLenOp, ReadScalarOp, ShapeOp, StrideOp};
use cubecl_core::ir::dialect::math::IAddOp;
use cubecl_core::ir::metadata::Info;
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::ElemType;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::TypeAttr;
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron_llvm::op_interfaces::CastOpWithNNegInterface;
use pliron_llvm::ops as llvm;
use pliron_llvm::types::PointerType as LlvmPointerType;
use std::rc::Rc;

use crate::compiler::shared_memory::{SharedDeclarations, SharedMemories};
use crate::compiler::to_llvm::ty::{INDEX_WIDTH, cube_type_to_llvm};

/// `(op, buffer_idx, result)` for each `cube.buffer_len`, gathered during the walk so the ops
/// can be rewritten once the walker no longer holds them borrowed.
#[derive(Default)]
struct BufferLens(Vec<(Ptr<Operation>, usize, Value)>);

/// `(op, elem_ty, id, result)` for each `cube.read_scalar`, gathered during the walk so the ops
/// can be rewritten once the walker no longer holds them borrowed.
#[derive(Default)]
struct ReadScalars(Vec<(Ptr<Operation>, TypeHandle, usize, Value)>);

/// `(op, buffer_idx, dim, result)` for each `cube.shape` and `cube.stride`. Both read
/// `dynamic_meta[static_meta[slot] + dim]`, so only the static slot differs: stride offsets are
/// pre-biased by the host past the shapes region.
#[derive(Default)]
struct DynMetaReads(Vec<(Ptr<Operation>, usize, Value, Value)>);

/// Collapses buffer args and shared memories behind `%buffer_ptrs`, lowers `cube.buffer_len`,
/// `cube.read_scalar`, `cube.shape` and `cube.stride` against `%metadata`. The info buffer is
/// laid out as `[scalars | static meta | dynamic meta]`, so scalar reads index straight from its
/// front while metadata reads must skip the scalar prefix.
pub struct LowerEntryAbiPass {
    info: Info,
    /// Filled in with the shared memory the host must reserve, see [`SharedMemories`].
    shared_memories: Rc<RefCell<SharedMemories>>,
}

impl LowerEntryAbiPass {
    pub fn new(info: Info, shared_memories: Rc<RefCell<SharedMemories>>) -> Self {
        Self {
            info,
            shared_memories,
        }
    }
}

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

        let shared = SharedDeclarations::collect(ctx, op);

        let mut buffer_lens = BufferLens::default();
        visit_all_ops_of_type::<BufferLenOp, _>(ctx, &mut buffer_lens, op, |ctx, state, bl| {
            state
                .0
                .push((bl.get_operation(), bl.buffer_idx(ctx).0, bl.get_result(ctx)));
        });

        let mut read_scalars = ReadScalars::default();
        visit_all_ops_of_type::<ReadScalarOp, _>(ctx, &mut read_scalars, op, |ctx, state, rs| {
            state.0.push((
                rs.get_operation(),
                rs.ty(ctx).get_type(ctx),
                rs.id(ctx).0,
                rs.get_result(ctx),
            ));
        });

        let mut shapes = DynMetaReads::default();
        visit_all_ops_of_type::<ShapeOp, _>(ctx, &mut shapes, op, |ctx, state, sh| {
            state.0.push((
                sh.get_operation(),
                sh.buffer_idx(ctx).0,
                sh.dim(ctx),
                sh.get_result(ctx),
            ));
        });

        let mut strides = DynMetaReads::default();
        visit_all_ops_of_type::<StrideOp, _>(ctx, &mut strides, op, |ctx, state, st| {
            state.0.push((
                st.get_operation(),
                st.buffer_idx(ctx).0,
                st.dim(ctx),
                st.get_result(ctx),
            ));
        });

        let dyn_meta_reads: Vec<_> =
            (shapes.0.iter())
                .map(|(op, idx, dim, res)| (*op, self.info.shape_offset_index(*idx), *dim, *res))
                .chain((strides.0.iter()).map(|(op, idx, dim, res)| {
                    (*op, self.info.stride_offset_index(*idx), *dim, *res)
                }))
                .collect();

        let ptr_ty: TypeHandle = LlvmPointerType::get(ctx, 0).into();
        let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let index_ty: TypeHandle = IntegerType::get(ctx, INDEX_WIDTH, Signedness::Signless).into();
        // `cube.index` always lowers to `i64`, but the host stores anything index-shaped —
        // metadata slots and `usize` scalars — at the kernel's address width.
        let addr_width = 8 * ctx.address_type().size() as u32;
        let slot_ty: TypeHandle = IntegerType::get(ctx, addr_width, Signedness::Signless).into();

        let meta_idx = BasicBlock::push_argument(entry, ctx, ptr_ty);
        let meta_ptr = entry.deref(ctx).get_argument(meta_idx);

        // The metadata regions sit behind the scalar prefix, so byte-step past it first.
        let static_offset = self.info.sized_meta.map_or(0, |field| field.offset);
        for (bl_op, buffer_idx, result) in &buffer_lens.0 {
            let slot = self.info.buffer_len_index(*buffer_idx);
            let base = byte_offset(ctx, meta_ptr, static_offset, i8_ty, *bl_op);
            let len = load_slot(
                ctx,
                base,
                llvm::GepIndex::Constant(slot as u32),
                slot_ty,
                index_ty,
                *bl_op,
            );
            result.replace_all_uses_with(ctx, &len);
            Operation::erase(*bl_op, ctx);
        }

        for (read_op, slot, dim, result) in &dyn_meta_reads {
            let static_base = byte_offset(ctx, meta_ptr, static_offset, i8_ty, *read_op);
            let tensor_offset = load_slot(
                ctx,
                static_base,
                llvm::GepIndex::Constant(*slot as u32),
                slot_ty,
                index_ty,
                *read_op,
            );
            let idx = IAddOp::new(ctx, tensor_offset, *dim);
            idx.get_operation().insert_before(ctx, *read_op);

            let dyn_base = byte_offset(
                ctx,
                meta_ptr,
                self.info.dynamic_meta_offset,
                i8_ty,
                *read_op,
            );
            let value = load_slot(
                ctx,
                dyn_base,
                llvm::GepIndex::Value(idx.get_result(ctx)),
                slot_ty,
                index_ty,
                *read_op,
            );
            result.replace_all_uses_with(ctx, &value);
            Operation::erase(*read_op, ctx);
        }

        for (rs_op, elem_ty, id, result) in &read_scalars.0 {
            let field = self
                .info
                .scalars
                .iter()
                .find(|field| field.ty.to_type(ctx) == *elem_ty)
                .unwrap_or_else(|| panic!("cube.read_scalar has no matching scalar in the info"));
            let stored_ty = match field.ty {
                ElemType::Index => slot_ty,
                _ => cube_type_to_llvm(ctx, *elem_ty),
            };
            let offset = field.offset;

            // Byte-step to the type group's base, then step `id` elements into it.
            let group = byte_offset(ctx, meta_ptr, offset, i8_ty, *rs_op);
            let scalar = load_slot(
                ctx,
                group,
                llvm::GepIndex::Constant(*id as u32),
                stored_ty,
                cube_type_to_llvm(ctx, *elem_ty),
                *rs_op,
            );
            result.replace_all_uses_with(ctx, &scalar);
            Operation::erase(*rs_op, ctx);
        }

        // Collapse buffers and shared memories behind a single leading `%buffer_ptrs`. The
        // shared memories take the slots after the last buffer.
        let shared_base = (buffers.iter())
            .map(|(_, buffer_pos, _)| buffer_pos + 1)
            .max()
            .unwrap_or(0);
        if !buffers.is_empty() || !shared.is_empty() {
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

            let blocks = shared.lower(ctx, buffer_ptrs, shared_base, terminator);
            if !blocks.is_empty() {
                *self.shared_memories.borrow_mut() = SharedMemories {
                    base: shared_base,
                    blocks,
                };
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

/// Byte-step `bytes` past `ptr`, inserting the walk before `before`.
fn byte_offset(
    ctx: &mut Context,
    ptr: Value,
    bytes: usize,
    i8_ty: TypeHandle,
    before: Ptr<Operation>,
) -> Value {
    let gep = llvm::GetElementPtrOp::new(
        ctx,
        ptr,
        vec![llvm::GepIndex::Constant(bytes as u32)],
        i8_ty,
    );
    gep.get_operation().insert_before(ctx, before);
    gep.get_result(ctx)
}

/// Load the `stored_ty` slot at `index` from `base`, widening it to the `result_ty` the kernel
/// expects. The two only differ for index-shaped values under 32-bit addressing.
fn load_slot(
    ctx: &mut Context,
    base: Value,
    index: llvm::GepIndex,
    stored_ty: TypeHandle,
    result_ty: TypeHandle,
    before: Ptr<Operation>,
) -> Value {
    let gep = llvm::GetElementPtrOp::new(ctx, base, vec![index], stored_ty);
    gep.get_operation().insert_before(ctx, before);
    let load = llvm::LoadOp::new(ctx, gep.get_result(ctx), stored_ty);
    load.get_operation().insert_before(ctx, before);

    if stored_ty == result_ty {
        return load.get_result(ctx);
    }
    let zext = llvm::ZExtOp::new_with_nneg(ctx, load.get_result(ctx), result_ty, true);
    zext.get_operation().insert_before(ctx, before);
    zext.get_result(ctx)
}
