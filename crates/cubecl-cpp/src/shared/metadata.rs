
use cubecl_core::ir::{
    CanMaterialize, Pure,
    attributes::{ATTR_READONLY, FuncInterface, IndexAttr},
    dialect::general::{BufferLenOp, ReadScalarOp, ShapeOp, StrideOp},
    ident,
    prelude::*,
    rewrite::RewriteOp,
    types::scalar::IndexType,
};
use pliron::{
    builtin::{attributes::TypeAttr, ops::FuncOp},
    debug_info::{insert_block_arg_name, insert_operation_result_name},
    irbuild::match_rewrite::apply_match_rewrite,
};

use crate::{
    cuda::signature::ATTR_GRID_CONSTANT,
    shared::{
        CompilationOptions, CompilationState, shared_op_with_out,
        signature::{LoadDynMetaOp, LoadInfoOp},
        ty::{InfoStructType, TypeExtCPP, UniformPointerType},
    },
};

#[cube_op(name = "cpp.read_scalar", format = "$0 `[` attr($id, $IndexAttr) `]`")]
#[result_ty(from_inputs = |ctx, _, _, ty: &TypeAttr| ty.get_type(ctx))]
#[op_traits(Pure, CanMaterialize)]
pub struct CppReadScalarOp {
    pub base: Value,
    pub id: IndexAttr,
    pub ty: TypeAttr,
}

#[cube_op(
    name = "cpp.read_static_meta",
    format = "$0 `[` attr($offset, $IndexAttr) `]`"
)]
#[result_ty(fixed = IndexType::get(ctx).to_handle())]
#[op_traits(Pure, CanMaterialize)]
pub struct CppReadStaticMetaOp {
    pub base: Value,
    pub offset: IndexAttr,
}

#[cube_op(name = "cpp.read_dynamic_meta", format = "$0 `[` $1 ` + ` $2 `]`")]
#[result_ty(fixed = IndexType::get(ctx).to_handle())]
#[op_traits(Pure, CanMaterialize)]
pub struct CppReadDynamicMetaOp {
    pub base: Value,
    pub offset: Value,
    pub dim: Value,
}

shared_op_with_out!(CppReadScalarOp, |op, ctx| {
    let base = op.base(ctx).name(ctx);
    let elem = op.ty(ctx).to_cpp(ctx);
    let offset = op.id(ctx).0;
    format!("{base}.scalars_{elem}[{offset}]")
});

shared_op_with_out!(CppReadStaticMetaOp, |op, ctx| {
    let base = op.base(ctx).name(ctx);
    let offset = op.offset(ctx).0;
    format!("{base}.static_meta[{offset}]")
});

shared_op_with_out!(CppReadDynamicMetaOp, |op, ctx| {
    let base = op.base(ctx).name(ctx);
    let offset = op.offset(ctx).name(ctx);
    let dim = op.dim(ctx).name(ctx);
    format!("{base}[{offset} + {dim}]")
});

#[derive(Default)]
pub struct LowerInfoPass;

#[pass_name]
impl Pass for LowerInfoPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let (has_info, has_dynamic_meta) = {
            let info = &ctx.aux_ty::<CompilationState>().info;
            (info.has_info(), info.has_dynamic_meta)
        };
        let func = op.as_op::<FuncOp>(ctx).unwrap();
        let entry_block = func.get_entry_block(ctx);
        let supports_features = ctx.aux_ty::<CompilationOptions>().supports_features;

        let info_name = ident("info");
        let dyn_meta_name = ident("dynamic_meta");

        let mut info_st = None;
        let mut dyn_meta = None;

        if supports_features.grid_constants {
            if has_dynamic_meta {
                let usize = IndexType::get(ctx).to_handle();
                let index_ptr = UniformPointerType::get(ctx, usize);
                let id = func.push_argument(ctx, index_ptr.to_handle());
                func.set_arg_attr_unit(ctx, id, &ATTR_READONLY);
                insert_block_arg_name(ctx, entry_block, id, Some(dyn_meta_name));
                let value = entry_block.deref(ctx).get_argument(id);
                dyn_meta = Some(value);
            }

            if has_info {
                let id = func.push_argument(ctx, InfoStructType::get(ctx).into());
                func.set_arg_attr_unit(ctx, id, &ATTR_GRID_CONSTANT);
                func.set_arg_attr_unit(ctx, id, &ATTR_READONLY);
                insert_block_arg_name(ctx, entry_block, id, Some(info_name));
                let value = entry_block.deref(ctx).get_argument(id);
                info_st = Some(value);
            }
        } else if has_info {
            let info_st_ty = InfoStructType::get(ctx).to_handle();
            let info_ptr = UniformPointerType::get(ctx, info_st_ty);
            let id = func.push_argument(ctx, info_ptr.to_handle());
            func.set_arg_attr_unit(ctx, id, &ATTR_READONLY);

            let ptr = entry_block.deref(ctx).get_argument(id);

            let load_info = LoadInfoOp::new(ctx, ptr);
            info_st = Some(load_info.get_result(ctx));
            insert_operation_result_name(ctx, load_info.get_operation(), 0, Some(info_name));
            load_info.get_operation().insert_at_front(entry_block, ctx);

            let load_dyn = LoadDynMetaOp::new(ctx, ptr);
            dyn_meta = Some(load_dyn.get_result(ctx));
            insert_operation_result_name(ctx, load_dyn.get_operation(), 0, Some(dyn_meta_name));
            load_dyn.get_operation().insert_at_front(entry_block, ctx);
        }

        if let Some(info_st) = info_st {
            let mut rewrite = MatchRewriteOp::new(ReplaceScalars(info_st));
            apply_match_rewrite(ctx, &mut rewrite, Default::default(), op)?;
            let mut rewrite = MatchRewriteOp::new(ReplaceBufferLen(info_st));
            apply_match_rewrite(ctx, &mut rewrite, Default::default(), op)?;
        }
        if let Some((info_st, dyn_meta)) = info_st.zip(dyn_meta) {
            let mut rewrite = MatchRewriteOp::new(ReplaceShape(info_st, dyn_meta));
            apply_match_rewrite(ctx, &mut rewrite, Default::default(), op)?;
            let mut rewrite = MatchRewriteOp::new(ReplaceStride(info_st, dyn_meta));
            apply_match_rewrite(ctx, &mut rewrite, Default::default(), op)?;
        }

        let mut res = PassResult::default();
        res.ir_changed |= IRStatus::Changed;
        Ok(res)
    }
}

struct ReplaceScalars(Value);
impl RewriteOp<ReadScalarOp> for ReplaceScalars {
    fn rewrite(&mut self, ctx: &mut Context, rewriter: &mut MatchRewriter, op: ReadScalarOp) {
        let id = *op.id(ctx);
        let ty = op.ty(ctx).clone();
        let new_op = CppReadScalarOp::new(ctx, self.0, id, ty);
        rewriter.replace_op_with(ctx, op.get_operation(), new_op.get_operation());
    }
}

struct ReplaceBufferLen(Value);
impl RewriteOp<BufferLenOp> for ReplaceBufferLen {
    fn rewrite(&mut self, ctx: &mut Context, rewriter: &mut MatchRewriter, op: BufferLenOp) {
        let buffer_idx = op.buffer_idx(ctx).0;
        let meta = ctx.aux_ty::<CompilationState>().info.metadata;
        let offset = meta.buffer_len_index(buffer_idx);
        let new_op = CppReadStaticMetaOp::new(ctx, self.0, offset);
        rewriter.replace_op_with(ctx, op.get_operation(), new_op.get_operation());
    }
}

struct ReplaceShape(Value, Value);
impl RewriteOp<ShapeOp> for ReplaceShape {
    fn rewrite(&mut self, ctx: &mut Context, rewriter: &mut MatchRewriter, op: ShapeOp) {
        let (static_, dynamic) = (self.0, self.1);
        let buffer_idx = op.buffer_idx(ctx).0;
        let meta = ctx.aux_ty::<CompilationState>().info.metadata;
        let offset = meta.shape_offset_index(buffer_idx);
        let dim = op.dim(ctx);

        let offs = CppReadStaticMetaOp::new(ctx, static_, offset);
        offs.get_operation().insert_before(ctx, op.get_operation());
        let new_op = CppReadDynamicMetaOp::new(ctx, dynamic, offs.get_result(ctx), dim);
        rewriter.replace_op_with(ctx, op.get_operation(), new_op.get_operation());
    }
}

struct ReplaceStride(Value, Value);
impl RewriteOp<StrideOp> for ReplaceStride {
    fn rewrite(&mut self, ctx: &mut Context, rewriter: &mut MatchRewriter, op: StrideOp) {
        let (static_, dynamic) = (self.0, self.1);
        let buffer_idx = op.buffer_idx(ctx).0;
        let meta = ctx.aux_ty::<CompilationState>().info.metadata;
        let offset = meta.stride_offset_index(buffer_idx);
        let dim = op.dim(ctx);

        let offs = CppReadStaticMetaOp::new(ctx, static_, offset);
        offs.get_operation().insert_before(ctx, op.get_operation());
        let new_op = CppReadDynamicMetaOp::new(ctx, dynamic, offs.get_result(ctx), dim);
        rewriter.replace_op_with(ctx, op.get_operation(), new_op.get_operation());
    }
}
