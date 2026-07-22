use cubecl_common::stub::LazyLock;
use cubecl_ir::{
    AddressSpace, Scope,
    attributes::{BufferBindingAttr, BufferIOAttr, IndexAttr},
    dialect::{
        general::{BufferLenOp, ReadScalarOp, ShapeOp, StrideOp},
        math::IAddOp,
    },
    ident,
    metadata::Info,
    prelude::*,
    types::{ArrayType, RuntimeArrayType, scalar::IndexType},
};
use itertools::Itertools;
use pliron::{
    attribute::AttrObj,
    builtin::{attributes::VecAttr, ops::FuncOp},
    identifier::Identifier,
};

use crate::compiler::wgsl::{
    GlobalVariableOp,
    lower::LowerOp,
    to_wgsl::{OpToWgsl, TypeExtWgsl, wgsl_op_with_out},
    types::StructType,
};

pub static INFO_ST: LazyLock<Identifier> = LazyLock::new(|| "info_st".try_into().unwrap());
pub static INFO_VAR: LazyLock<Identifier> = LazyLock::new(|| "info_var".try_into().unwrap());
pub static STATIC_META: LazyLock<Identifier> = LazyLock::new(|| "static_meta".try_into().unwrap());
pub static DYNAMIC_META: LazyLock<Identifier> =
    LazyLock::new(|| "dynamic_meta".try_into().unwrap());

wgsl_op_with_out!(ReadScalarOp; |op, ctx| {
    let ty = op.ty(ctx).to_wgsl(ctx);
    format!("{}.scalars_{ty}[{}]", &*INFO_VAR, op.id(ctx).0)
});

#[cube_op(name = "wgsl.read_static_meta")]
#[result_ty(fixed = IndexType::get(ctx).to_handle())]
pub struct ReadStaticMetaOp {
    pub idx: IndexAttr,
}

wgsl_op_with_out!(ReadStaticMetaOp; |op, ctx| {
    let field = &*STATIC_META;
    format!("{}.{field}[{}]", &*INFO_VAR, op.idx(ctx).0)
});

#[cube_op(name = "wgsl.read_dynamic_meta")]
#[result_ty(fixed = IndexType::get(ctx).to_handle())]
pub struct ReadDynamicMetaOp {
    pub idx: Value,
}

wgsl_op_with_out!(ReadDynamicMetaOp; |op, ctx| {
    let field = &*DYNAMIC_META;
    format!("{}.{field}[{}]", &*INFO_VAR, op.idx(ctx).name(ctx))
});

#[pliron_attr(name = "wgsl.field", format = "`@` $name `: ` $ty", verifier = "succ")]
#[derive(Debug, Clone, PartialEq, Eq, new)]
pub struct FieldAttr {
    pub name: Identifier,
    pub ty: TypeHandle,
}

#[pliron_op(name = "wgsl.def_struct", format, attributes = (wgsl_def_struct_fields: VecAttr), verifier = "succ")]
#[derive_op_interface_impl(SymbolOpInterface)]
pub struct StructDefOp;

impl StructDefOp {
    pub fn new(ctx: &mut Context, name: Identifier, fields: Vec<FieldAttr>) -> Self {
        let fields = fields.into_iter().map(|it| -> AttrObj { Box::new(it) });
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        };
        op.set_symbol_name(ctx, name);
        op.set_attr_wgsl_def_struct_fields(ctx, VecAttr(fields.collect()));
        op
    }

    pub fn fields(&self, ctx: &Context) -> Vec<FieldAttr> {
        let attr = self.get_attr_wgsl_def_struct_fields(ctx).unwrap();
        attr.0
            .iter()
            .map(|it| it.downcast_ref::<FieldAttr>().unwrap().clone())
            .collect()
    }
}

#[op_interface_impl]
impl OpToWgsl for StructDefOp {
    fn to_wgsl(&self, ctx: &Context) -> String {
        let name = self.get_symbol_name(ctx);
        let fields = self.fields(ctx).into_iter();
        let mut fields = fields.map(|field| format!("{}: {}", field.name, field.ty.to_wgsl(ctx)));
        format!("struct {name} {{ {} }}\n", fields.join(", "))
    }
}

pub fn declare_info(ctx: &mut Context, entry_func: FuncOp, num_buffers: usize) {
    let info = ctx.aux_ty::<Info>();
    let entry = entry_func.get_operation();

    if !info.has_info() {
        return;
    }

    let mut fields = vec![];
    for scalar in &info.scalars {
        let elem_ty = scalar.ty.to_type(ctx);
        let name = ident(format!("scalars_{}", elem_ty.to_wgsl(ctx)));
        let ty = ArrayType::get(ctx, elem_ty, scalar.padded_size(ctx)).to_handle();
        fields.push(FieldAttr::new(name, ty));
    }
    if let Some(field) = info.sized_meta {
        let ty = ArrayType::get(ctx, IndexType::get(ctx).into(), field.padded_size(ctx));
        fields.push(FieldAttr::new(STATIC_META.clone(), ty.into()));
    }
    if info.has_dynamic_meta {
        let ty = RuntimeArrayType::get(ctx, IndexType::get(ctx).into());
        fields.push(FieldAttr::new(DYNAMIC_META.clone(), ty.into()));
    }
    let struct_ = StructDefOp::new(ctx, INFO_ST.clone(), fields);
    struct_.get_operation().insert_before(ctx, entry);

    let var_ty = StructType::get(ctx, INFO_ST.clone()).to_handle();
    let binding = BufferBindingAttr::new(num_buffers, None);
    let var = GlobalVariableOp::new(
        ctx,
        var_ty,
        AddressSpace::Global(0),
        Some(binding),
        Some(BufferIOAttr::ReadOnly),
    );
    var.set_symbol_name(ctx, INFO_VAR.clone());

    var.get_operation()
        .insert_after(ctx, struct_.get_operation());
}

#[op_interface_impl]
impl LowerOp for BufferLenOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let buffer_idx = self.buffer_idx(ctx).0;
        let id = ctx.aux_ty::<Info>().buffer_len_index(buffer_idx);
        let new_op = ReadStaticMetaOp::new(scope.ctx_mut(), id);
        vec![scope.register_with_result(&new_op)]
    }
}

#[op_interface_impl]
impl LowerOp for ShapeOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx_mut();
        let buffer_idx = self.buffer_idx(ctx).0;
        let dim = self.dim(ctx);
        let id = ctx.aux_ty::<Info>().shape_offset_index(buffer_idx);
        let base_offs = scope.register_with_result(&ReadStaticMetaOp::new(scope.ctx_mut(), id));
        let offset = scope.register_with_result(&IAddOp::new(scope.ctx_mut(), base_offs, dim));
        vec![scope.register_with_result(&ReadDynamicMetaOp::new(ctx, offset))]
    }
}

#[op_interface_impl]
impl LowerOp for StrideOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx_mut();
        let buffer_idx = self.buffer_idx(ctx).0;
        let dim = self.dim(ctx);
        let id = ctx.aux_ty::<Info>().stride_offset_index(buffer_idx);
        let base_offs = scope.register_with_result(&ReadStaticMetaOp::new(scope.ctx_mut(), id));
        let offset = scope.register_with_result(&IAddOp::new(scope.ctx_mut(), base_offs, dim));
        vec![scope.register_with_result(&ReadDynamicMetaOp::new(ctx, offset))]
    }
}
