//! Kernel arguments and metadata.

use crate::prelude::*;
use cubecl_core::ir::{
    ElemType,
    attributes::{ATTR_BUFFER_BINDING, BufferBindingAttr},
    dialect::{
        general::{BufferLenOp, CastOp, ReadScalarOp, ReinterpretCastOp, ShapeOp, StrideOp},
        math::IAddOp,
        memory::{IndexOp, LoadOp},
    },
    metadata::Info,
    types::{BytesType, PointerType, RuntimeArrayType},
};

#[derive(Default)]
struct BufferLens(Vec<(Ptr<Operation>, usize, Value)>);

#[derive(Default)]
struct ReadScalars(Vec<(Ptr<Operation>, TypeHandle, usize, Value)>);

#[derive(Default)]
struct DynMetaReads(Vec<(Ptr<Operation>, usize, Value, Value)>);

/// Whether scalars and static metadata use the kernel parameter block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridConstants(pub bool);

impl CtxGridConstants for Context {}

pub trait CtxGridConstants: ContextExt {
    fn grid_constants(&self) -> bool {
        self.aux_ty::<GridConstants>().0
    }
    fn set_grid_constants(&mut self, value: bool) {
        self.set_aux_ty(GridConstants(value));
    }
}

/// Host-visible kernel argument layout.
pub trait EntryArgLayout {
    fn present_args(
        &self,
        ctx: &mut Context,
        func: FuncOp,
        buffers: &[(usize, usize, Value)],
        shared: SharedDeclarations,
    );
}

/// Metadata layout: scalars, static metadata, then dynamic metadata.
/// With grid constants, dynamic metadata uses a separate buffer.
pub struct LowerEntryAbiPass {
    info: Info,
    layout: Box<dyn EntryArgLayout>,
}

impl LowerEntryAbiPass {
    pub fn new(info: Info, layout: Box<dyn EntryArgLayout>) -> Self {
        Self { info, layout }
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
        let grid_constants = ctx.grid_constants();
        let dyn_meta = (grid_constants && self.info.has_dynamic_meta).then(|| {
            let idx = BasicBlock::push_argument(entry, ctx, info_ty);
            entry.deref(ctx).get_argument(idx)
        });
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

        let (dyn_meta_ptr, dyn_base) = match dyn_meta {
            Some(dyn_meta) => (dyn_meta, 0),
            None => (info, elem_index(self.info.dynamic_meta_offset, slot_size)),
        };
        for (read_op, slot, dim, result) in &dyn_meta_reads {
            let slot = index_const(ctx, static_base + *slot, *read_op);
            let tensor_offset = load_info(ctx, info, slot_ty, slot, *read_op);
            let tensor_offset = to_index(ctx, tensor_offset, *read_op);

            let index = index_add(ctx, tensor_offset, *dim, *read_op);
            let dyn_base = index_const(ctx, dyn_base, *read_op);
            let index = index_add(ctx, index, dyn_base, *read_op);
            let value = load_info(ctx, dyn_meta_ptr, slot_ty, index, *read_op);
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

        self.layout.present_args(ctx, func, &buffers, shared);

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

pub(crate) fn rebuild_func_type(ctx: &mut Context, func: FuncOp) {
    let entry = func.get_entry_block(ctx);
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
}

fn ptr_to(ctx: &Context, inner: TypeHandle) -> TypeHandle {
    PointerType::get(ctx, inner, AddressSpace::Global(0)).into()
}

fn array_ptr_to(ctx: &Context, elem: TypeHandle) -> TypeHandle {
    ptr_to(ctx, RuntimeArrayType::get(ctx, elem).into())
}

pub(crate) fn table_ty(ctx: &Context) -> TypeHandle {
    array_ptr_to(ctx, ptr_to(ctx, BytesType::get(ctx).into()))
}

/// Metadata regions are aligned to `INFO_ALIGN`.
fn elem_index(bytes: usize, elem_size: usize) -> usize {
    debug_assert_eq!(
        bytes % elem_size,
        0,
        "info regions are aligned, so they start on an element boundary"
    );
    bytes / elem_size
}

fn index_const(ctx: &mut Context, value: usize, before: Ptr<Operation>) -> Value {
    let constant = ConstantOp::new(ctx, Box::new(IndexAttr::new(value)));
    constant.get_operation().insert_before(ctx, before);
    constant.get_result(ctx)
}

fn index_add(ctx: &mut Context, lhs: Value, rhs: Value, before: Ptr<Operation>) -> Value {
    let add = IAddOp::new(ctx, lhs, rhs);
    add.get_operation().insert_before(ctx, before);
    add.get_result(ctx)
}

fn to_index(ctx: &mut Context, value: Value, before: Ptr<Operation>) -> Value {
    let index_ty = IndexType::get(ctx).into();
    let cast = CastOp::new(ctx, index_ty, value);
    cast.get_operation().insert_before(ctx, before);
    cast.get_result(ctx)
}

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

fn load_elem(ctx: &mut Context, base: Value, index: Value, before: Ptr<Operation>) -> Value {
    let elem = IndexOp::new(ctx, base, index, None);
    elem.get_operation().insert_before(ctx, before);
    let load = LoadOp::new(ctx, elem.get_result(ctx));
    load.get_operation().insert_before(ctx, before);
    load.get_result(ctx)
}

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
