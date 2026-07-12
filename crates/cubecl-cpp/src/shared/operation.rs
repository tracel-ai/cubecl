use cubecl_core::{
    cmma::MatrixType,
    ir::{
        AddressSpace, Scope, cube_op,
        dialect::{
            base::OperationPtrExt,
            general::{CommentOp, PoisonOp, PrintfOp},
            math::FmaOp,
            memory::{DeclareVariableOp, UnrelatedAllocInfo},
            synchronization::{SyncOp, SyncScope},
        },
        prelude::*,
        types::PointerType,
    },
};
use cubecl_opt::passes::alloc_shared_memory::SliceSharedOp;
use itertools::Itertools;
use pliron::{
    arg_err,
    attribute::AttrObj,
    builtin::{attributes::TypeAttr, ops::ConstantOp},
    opts::mem2reg::{AllocInfo, PromotableAllocationInterface},
};

use crate::{
    error::{CompileError, Result},
    shared::{CppValue, format_const, lowering::LowerOp, ty::TypeExtCPP},
    target::{Shared, dispatch_target},
};

#[op_interface]
pub trait OpToCPP<T> {
    verify_op_succ!();
    fn to_cpp(&self, ctx: &Context) -> String;
}

macro_rules! shared_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Shared> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(crate) use shared_op;

macro_rules! shared_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Shared> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                use cubecl_core::ir::prelude::*;
                use $crate::shared::CppValue;
                let op = $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl);
                let out = self.get_result(ctx).fmt_left(ctx);
                format!("{out} = {op};\n")
            }
        }
    };
}
pub(crate) use shared_op_with_out;

pub trait OpExtCPP {
    fn to_cpp(&self, ctx: &Context) -> Result<String>;
}

impl OpExtCPP for Ptr<Operation> {
    fn to_cpp(&self, ctx: &Context) -> Result<String> {
        let op_dyn = self.dyn_op(ctx);
        let target_cpp = dispatch_target!(ctx, {
            let inner_cpp = op_cast::<dyn OpToCPP<Target>>(op_dyn.as_ref());
            inner_cpp.map(|it| it.to_cpp(ctx))
        });
        if let Some(cpp) = target_cpp {
            return Ok(cpp);
        }
        let shared = op_cast::<dyn OpToCPP<Shared>>(op_dyn.as_ref())
            .ok_or_else(|| CompileError::UnsupportedOp(op_dyn.disp(ctx).to_string()))?;
        Ok(shared.to_cpp(ctx))
    }
}

#[cube_op(name = "cpp.declare_local")]
#[result_ty(from_inputs = variable_ptr_ty)]
pub struct DeclareLocalOp {
    pub value_ty: TypeAttr,
    #[attribute(optional, untyped)]
    pub initializer: AttrObj,
}

#[op_interface_impl]
impl PromotableAllocationInterface for DeclareLocalOp {
    fn alloc_info(&self, ctx: &Context) -> Vec<AllocInfo> {
        vec![AllocInfo {
            ptr: self.get_result(ctx),
            ty: self.value_ty(ctx).get_type(ctx),
        }]
    }

    fn default_value(
        &self,
        ctx: &mut Context,
        inserter: &mut dyn Inserter,
        alloc_info: &AllocInfo,
    ) -> cubecl_core::ir::prelude::Result<Value> {
        if alloc_info.ptr != self.get_result(ctx) {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        if let Some(initializer) = self.initializer(ctx).map(|it| it.clone()) {
            let constant = ConstantOp::new(ctx, initializer);
            inserter.insert_op(ctx, &constant);
            Ok(constant.get_result(ctx))
        } else {
            let poison = PoisonOp::new(ctx, alloc_info.ty);
            inserter.insert_op(ctx, &poison);
            Ok(poison.get_result(ctx))
        }
    }

    fn promote(
        &self,
        ctx: &mut Context,
        rewriter: &mut dyn Rewriter,
        alloc_infos: &[AllocInfo],
    ) -> cubecl_core::ir::prelude::Result<()> {
        if alloc_infos.len() != 1 || alloc_infos[0].ptr != self.get_result(ctx) {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

shared_op!(DeclareLocalOp, |op, ctx| {
    let ty = op.value_ty(ctx).get_type(ctx);
    let name = op.get_result(ctx).name(ctx);
    let value_ty = ty.to_cpp(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    let init = op
        .initializer(ctx)
        .map(|init| format_const(ctx, init.clone(), ty));
    if let Some(init) = init {
        format!("{value_ty} {name}_store = {init};\n{out_ty} {name} = &{name}_store;")
    } else {
        format!("{value_ty} {name}_store;\n{out_ty} {name} = &{name}_store;")
    }
});

#[cube_op(name = "cpp.declare_matrix")]
#[result_ty(from_inputs = variable_ptr_ty)]
pub struct DeclareMatrixOp {
    pub value_ty: TypeAttr,
}

fn variable_ptr_ty(ctx: &Context, value_ty: &TypeAttr) -> TypeHandle {
    let value_ty = value_ty.get_type(ctx);
    PointerType::get(ctx, value_ty, AddressSpace::Local).into()
}

shared_op_with_out!(SliceSharedOp, |op, ctx| {
    let ptr_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    let block = op.block(ctx).name(ctx);
    let offset = op.offset(ctx).0;
    format!("reinterpret_cast<{ptr_ty}>(&{block}[{offset}])")
});

// These are the same in IR semantics, but have very different codegen in C++. So split them up.
#[op_interface_impl]
impl LowerOp for DeclareVariableOp {
    fn should_lower(&self, _ctx: &Context) -> bool {
        true
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx_mut();
        let value_ty = self.value_ty(ctx).clone();
        let addr_space = self.addr_space(ctx).0;
        vec![match addr_space {
            AddressSpace::Global(_) => panic!("Unsupported address space for declaration"),
            AddressSpace::Shared => panic!("Should be lowered to block allocation"),
            AddressSpace::Local => {
                if value_ty.get_type(ctx).deref(ctx).is::<MatrixType>() {
                    let op = DeclareMatrixOp::new(ctx, value_ty);
                    scope.register_with_result(&op)
                } else {
                    let init = self.initializer(ctx).map(|it| it.clone());
                    let op = DeclareLocalOp::new(ctx, value_ty, init);
                    scope.register_with_result(&op)
                }
            }
        }]
    }
}

shared_op_with_out!(FmaOp, |op, ctx| {
    let a = op.a(ctx).name(ctx);
    let b = op.b(ctx).name(ctx);
    let c = op.c(ctx).name(ctx);
    format!("fma({a}, {b}, {c});")
});

shared_op!(CommentOp, |op, ctx| {
    let content = String::from(op.comment(ctx).clone());
    if content.contains("\n") {
        format!("/* {content} */\n")
    } else {
        format!("// {content}\n")
    }
});

shared_op!(PrintfOp, |op, ctx| {
    let format_string = op.format_string(ctx);
    let args = op.args(ctx);
    let args = args.iter().map(|it| format!(", {}", it.name(ctx))).join("");
    format!("printf({:?}{args});", format_string.as_str())
});

shared_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane => "__syncwarp();\n",
        SyncScope::Cube | SyncScope::Device => "__syncthreads();\n",
    }
    .into()
});
