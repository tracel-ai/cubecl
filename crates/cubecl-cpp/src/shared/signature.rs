use core::any::type_name;

use cubecl_core::ir::{
    ContextExt,
    attributes::FuncInterface,
    cube_op,
    dialect::OperationPtrExt,
    ident,
    interfaces::AlignedType,
    prelude::*,
    types::{VectorType, scalar::IndexType},
};
use cubecl_runtime::kernel::Visibility;
use itertools::Itertools;
use pliron::{
    builtin::{
        attributes::{TypeAttr, UnitAttr},
        op_interfaces::SingleBlockRegionInterface,
        ops::{FuncOp, ModuleOp},
    },
    debug_info::{insert_block_arg_name, insert_operation_result_name},
    deps::hash::FxHashMap,
    graph::walkers::uninterruptible::immutable::walk_op,
    pass_manager::Pass,
};

use crate::{
    cuda::signature::ATTR_GRID_CONST,
    shared::{
        ATTR_CONST, ATTR_RESTRICT, CompilationOptions, CompilationState, CppValue,
        branch::block_to_cpp,
        shared_op, shared_op_with_out,
        ty::{AddressSpace, InfoStructType, PointerType, TypeExtCPP},
        type_definitions, type_info_definition_sized,
    },
};

shared_op!(ModuleOp, |op, ctx| {
    let state = ctx.aux_ty::<CompilationState>();
    let mut out = String::new();
    type_definitions(&mut out, ctx).unwrap();
    type_info_definition_sized(&mut out, ctx, &state.info).unwrap();
    out.push_str(&block_to_cpp(ctx, op.get_body(ctx, 0)));
    out
});

#[cube_op(name = "cpp.load_info")]
#[result_ty(fixed = InfoStructType::get(ctx).into())]
pub struct LoadInfoOp {
    ptr: Value,
}

#[cube_op(name = "cpp.load_dynamic_meta")]
#[result_ty(fixed = PointerType::get(
    ctx,
    IndexType::get(ctx).into(),
    AddressSpace::Global(Visibility::Uniform)
).into())]
pub struct LoadDynMetaOp {
    ptr: Value,
}

shared_op!(LoadInfoOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let out = op.get_result(ctx);
    let out_ty = out.get_type(ctx).to_cpp(ctx);
    format!("const {out_ty}& {} = *{ptr};", out.name(ctx))
});

shared_op_with_out!(LoadDynMetaOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("reinterpret_cast<{out_ty}>({ptr} + 1)")
});

#[cube_op(name = "cpp.declare_vector")]
#[result_ty(none)]
pub struct DeclareVectorOp {
    vector_ty: TypeAttr,
}

shared_op!(DeclareVectorOp, |op, ctx| {
    let vector = op.vector_ty(ctx).get_type(ctx).deref(ctx);
    let vector = vector.downcast_ref::<VectorType>().unwrap();
    let align = vector.align(ctx);
    let inner_ty = vector.inner.to_cpp(ctx);
    let vec = vector.vectorization;
    let fields = (0..vec).map(|i| format!("{inner_ty} i_{i};")).join(" ");
    format!("struct __align__({align}) {inner_ty}_{vec} {{ {fields} }};\n")
});

#[derive(Default)]
pub struct LowerInfoPass;

impl Pass for LowerInfoPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &self,
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

        if supports_features.grid_constants {
            if has_dynamic_meta {
                let usize = IndexType::get(ctx).to_handle();
                let index_ptr =
                    PointerType::get(ctx, usize, AddressSpace::Global(Visibility::Uniform));
                let id = func.push_argument(ctx, index_ptr.to_handle());
                func.set_arg_attr(ctx, id, &ATTR_RESTRICT, Box::new(UnitAttr::new()));
                insert_block_arg_name(ctx, entry_block, id, Some(dyn_meta_name));
                let value = entry_block.deref(ctx).get_argument(id);
                ctx.aux_ty_mut::<CompilationState>().dynamic_meta = Some(value);
            }

            if has_info {
                let id = func.push_argument(ctx, InfoStructType::get(ctx).into());
                func.set_arg_attr(ctx, id, &ATTR_GRID_CONST, Box::new(UnitAttr::new()));
                func.set_arg_attr(ctx, id, &ATTR_CONST, Box::new(UnitAttr::new()));
                insert_block_arg_name(ctx, entry_block, id, Some(info_name));
                let value = entry_block.deref(ctx).get_argument(id);
                ctx.aux_ty_mut::<CompilationState>().info_st = Some(value);
            }
        } else if has_info {
            let info_st = InfoStructType::get(ctx).to_handle();
            let info_ptr =
                PointerType::get(ctx, info_st, AddressSpace::Global(Visibility::Uniform));
            let id = func.push_argument(ctx, info_ptr.to_handle());
            func.set_arg_attr(ctx, id, &ATTR_RESTRICT, Box::new(UnitAttr::new()));

            let ptr = entry_block.deref(ctx).get_argument(id);

            let load_info = LoadInfoOp::new(ctx, ptr);
            ctx.aux_ty_mut::<CompilationState>().info_st = Some(load_info.get_result(ctx));
            insert_operation_result_name(ctx, load_info.get_operation(), 0, Some(info_name));
            load_info.get_operation().insert_at_front(entry_block, ctx);

            let load_dyn = LoadDynMetaOp::new(ctx, ptr);
            ctx.aux_ty_mut::<CompilationState>().info_st = Some(load_dyn.get_result(ctx));
            insert_operation_result_name(ctx, load_dyn.get_operation(), 0, Some(dyn_meta_name));
            load_dyn.get_operation().insert_at_front(entry_block, ctx);
        }

        let mut res = PassResult::default();
        res.ir_changed |= IRStatus::Changed;
        Ok(res)
    }
}

/// Run on module
#[derive(Default)]
pub struct DeclareVectorTypesPass;

impl Pass for DeclareVectorTypesPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let module = op.as_op::<ModuleOp>(ctx).expect("Should be run on module");
        // Deduplicate by type name because some types are semantic only (i.e. tf32 is the same as f32)
        let mut vectors = FxHashMap::default();

        walk_op(
            ctx,
            &mut vectors,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, vectors, node| {
                let mut ins = |val: Value| {
                    let ty = val.get_type(ctx).deref(ctx);
                    if let Some(vector) = ty.downcast_ref::<VectorType>() {
                        vectors.insert((vector.inner.to_cpp(ctx), vector.vectorization), *vector);
                    }
                };
                match node {
                    IRNode::Operation(op) if let Some(res) = op.opt_result(ctx) => ins(res),
                    IRNode::BasicBlock(ptr) => {
                        for arg in ptr.deref(ctx).arguments() {
                            ins(arg);
                        }
                    }
                    _ => {}
                }
            },
        );

        let mut res = PassResult::default();
        for &vector in vectors.values() {
            let decl = DeclareVectorOp::new(ctx, TypeAttr::new(vector.get_self_handle(ctx)));
            decl.get_operation()
                .insert_at_front(module.get_body(ctx, 0), ctx);
            res.ir_changed |= IRStatus::Changed;
        }
        Ok(res)
    }
}
