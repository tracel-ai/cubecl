//! Rewrites the kernel entry ABI to what the JIT host calls.

use core::cell::RefCell;
use cubecl_core::ir::attributes::{
    ATTR_BUFFER_BINDING, BufferBindingAttr, FuncInterface, IndexAttr,
};
use cubecl_core::ir::dialect::general::{
    BufferLenOp, CastOp, ReadScalarOp, ReinterpretCastOp, ShapeOp, StrideOp,
};
use cubecl_core::ir::dialect::math::IAddOp;
use cubecl_core::ir::dialect::memory::{IndexOp, LoadOp};
use cubecl_core::ir::metadata::Info;
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::types::scalar::IndexType;
use cubecl_core::ir::types::{BytesType, PointerType, RuntimeArrayType};
use cubecl_core::ir::{AddressSpace, ElemType};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::TypeAttr;
use pliron::builtin::ops::{ConstantOp, FuncOp};
use pliron::builtin::types::FunctionType;
use std::rc::Rc;

use crate::shared::shared_memory::{SharedDeclarations, SharedMemories};
use crate::shared::to_llvm::ty::cube_type_to_llvm;

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

        let address_type = ctx.address_type();
        let slot_ty = address_type.unsigned_type().to_type(ctx);
        let slot_size = address_type.size();

        let info_ty = ptr_to(ctx, BytesType::get(ctx).into());
        let meta_idx = BasicBlock::push_argument(entry, ctx, info_ty);
        let info = entry.deref(ctx).get_argument(meta_idx);

        let static_base = elem_index(
            self.info.sized_meta.map_or(0, |field| field.offset),
            slot_size,
        );
        for (bl_op, buffer_idx, result) in &buffer_lens.0 {
            let slot = static_base + self.info.buffer_len_index(*buffer_idx);
            let slot = index_const(ctx, slot, *bl_op);
            let len = load_info(ctx, info, slot_ty, slot, *bl_op);
            let len = to_index(ctx, len, *bl_op);
            result.replace_all_uses_with(ctx, &len);
            Operation::erase(*bl_op, ctx);
        }

        let dyn_base = elem_index(self.info.dynamic_meta_offset, slot_size);
        for (read_op, slot, dim, result) in &dyn_meta_reads {
            let slot = index_const(ctx, static_base + *slot, *read_op);
            let tensor_offset = load_info(ctx, info, slot_ty, slot, *read_op);
            let tensor_offset = to_index(ctx, tensor_offset, *read_op);

            let index = index_add(ctx, tensor_offset, *dim, *read_op);
            let dyn_base = index_const(ctx, dyn_base, *read_op);
            let index = index_add(ctx, index, dyn_base, *read_op);
            let value = load_info(ctx, info, slot_ty, index, *read_op);
            let value = to_index(ctx, value, *read_op);
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
                _ => *elem_ty,
            };
            let stored_size = field.ty.expand_size(address_type);

            let index = elem_index(field.offset, stored_size) + *id;
            let index = index_const(ctx, index, *rs_op);
            let scalar = load_info(ctx, info, stored_ty, index, *rs_op);
            let scalar = match field.ty {
                ElemType::Index => to_index(ctx, scalar, *rs_op),
                _ => scalar,
            };
            result.replace_all_uses_with(ctx, &scalar);
            Operation::erase(*rs_op, ctx);
        }

        let shared_base = (buffers.iter())
            .map(|(_, buffer_pos, _)| buffer_pos + 1)
            .max()
            .unwrap_or(0);
        if !buffers.is_empty() || !shared.is_empty() {
            let table_ty = table_ty(ctx);
            BasicBlock::insert_argument(entry, ctx, 0, table_ty);
            let buffer_ptrs = entry.deref(ctx).get_argument(0);
            let terminator = entry
                .deref(ctx)
                .get_terminator(ctx)
                .expect("entry block must be terminated");

            for (_idx, buffer_pos, old_val) in &buffers {
                let buffer_ty = old_val.get_type(ctx);
                let buffer = load_table(ctx, buffer_ptrs, *buffer_pos, buffer_ty, terminator);
                old_val.replace_all_uses_with(ctx, &buffer);
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
        let arg_types: Vec<TypeHandle> = arg_values
            .iter()
            .map(|arg| cube_type_to_llvm(ctx, arg.get_type(ctx)))
            .collect();
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

/// `cube.ptr<inner>` into the memory the host owns.
fn ptr_to(ctx: &Context, inner: TypeHandle) -> TypeHandle {
    PointerType::get(ctx, inner, AddressSpace::Global(0)).into()
}

/// `cube.ptr<cube.runtime_array<elem>>`, the view `memory.index` walks to reach an element.
fn array_ptr_to(ctx: &Context, elem: TypeHandle) -> TypeHandle {
    ptr_to(ctx, RuntimeArrayType::get(ctx, elem).into())
}

/// Type of `%buffer_ptrs`, the table holding one pointer per buffer and then per shared memory.
/// What each of them points at is up to the slot's owner, so the table holds them opaquely.
fn table_ty(ctx: &Context) -> TypeHandle {
    array_ptr_to(ctx, ptr_to(ctx, BytesType::get(ctx).into()))
}

/// The element `bytes` bytes into an array of `elem_size` wide elements. The host aligns every
/// region of the info buffer to `INFO_ALIGN`, so no region starts inside an element.
fn elem_index(bytes: usize, elem_size: usize) -> usize {
    debug_assert_eq!(
        bytes % elem_size,
        0,
        "info regions are aligned, so they start on an element boundary"
    );
    bytes / elem_size
}

/// Insert a `cube.index` constant before `before`.
fn index_const(ctx: &mut Context, value: usize, before: Ptr<Operation>) -> Value {
    let constant = ConstantOp::new(ctx, Box::new(IndexAttr::new(value)));
    constant.get_operation().insert_before(ctx, before);
    constant.get_result(ctx)
}

/// Insert `lhs + rhs` before `before`, both being indices.
fn index_add(ctx: &mut Context, lhs: Value, rhs: Value, before: Ptr<Operation>) -> Value {
    let add = IAddOp::new(ctx, lhs, rhs);
    add.get_operation().insert_before(ctx, before);
    add.get_result(ctx)
}

/// Widen a stored slot to the `cube.index` the kernel computes with. The cast folds away when the
/// kernel addresses memory at 64 bits, where the two have the same width.
fn to_index(ctx: &mut Context, value: Value, before: Ptr<Operation>) -> Value {
    let index_ty = IndexType::get(ctx).into();
    let cast = CastOp::new(ctx, index_ty, value);
    cast.get_operation().insert_before(ctx, before);
    cast.get_result(ctx)
}

/// Load the element at `index` of the info buffer, read as an array of `elem_ty`.
fn load_info(
    ctx: &mut Context,
    info: Value,
    elem_ty: TypeHandle,
    index: Value,
    before: Ptr<Operation>,
) -> Value {
    let view_ty = array_ptr_to(ctx, elem_ty);
    let view = ReinterpretCastOp::new(ctx, view_ty, info);
    view.get_operation().insert_before(ctx, before);
    load_elem(ctx, view.get_result(ctx), index, before)
}

/// Load the element at `index` of the array `base` points at.
fn load_elem(ctx: &mut Context, base: Value, index: Value, before: Ptr<Operation>) -> Value {
    let elem = IndexOp::new(ctx, base, index, None);
    elem.get_operation().insert_before(ctx, before);
    let load = LoadOp::new(ctx, elem.get_result(ctx));
    load.get_operation().insert_before(ctx, before);
    load.get_result(ctx)
}

/// Read slot `slot` of the pointer table as a `ptr_ty`, which is how a buffer or a shared memory
/// reaches the block the host reserved for it. The table holds its pointers opaquely, so what a
/// slot holds is only known to its owner.
pub fn load_table(
    ctx: &mut Context,
    table: Value,
    slot: usize,
    ptr_ty: TypeHandle,
    before: Ptr<Operation>,
) -> Value {
    let slot = index_const(ctx, slot, before);
    let ptr = load_elem(ctx, table, slot, before);
    let cast = ReinterpretCastOp::new(ctx, ptr_ty, ptr);
    cast.get_operation().insert_before(ctx, before);
    cast.get_result(ctx)
}
