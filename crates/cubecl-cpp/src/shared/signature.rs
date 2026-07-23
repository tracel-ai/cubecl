use core::marker::PhantomData;

use cubecl_core::ir::{
    ContextExt, cube_op,
    dialect::OperationPtrExt,
    interfaces::{AlignedType, HasElementType},
    prelude::*,
    rewrite::visit_all_ops_of_type,
    types::{ArrayType, PointerType, RuntimeArrayType, VectorType, scalar::IndexType},
};
use cubecl_opt::passes::alloc_shared_memory::AllocSharedOp;
use itertools::Itertools;
use pliron::{
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::SingleBlockRegionInterface,
        ops::ModuleOp,
    },
    graph::walkers::uninterruptible::immutable::walk_op,
    pass::Pass,
    std_deps::hash::{FxHashMap, FxHashSet},
};

use crate::{
    shared::{
        CompilationState, CppValue,
        branch::block_to_cpp,
        shared_op, shared_op_with_out,
        ty::{InfoStructType, TypeExtCPP, UniformPointerType},
        type_definitions, type_info_definition_sized,
    },
    target::*,
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
#[result_ty(fixed = UniformPointerType::get(
    ctx,
    IndexType::get(ctx).into(),
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

#[cube_op(name = "cpp.declare_vector", format = "attr($vector_ty, $TypeAttr)")]
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

#[cube_op(name = "cpp.include", format = "attr($header, $StringAttr)")]
#[result_ty(none)]
pub struct IncludeOp {
    pub header: StringAttr,
}

shared_op!(IncludeOp, |op, ctx| {
    format!("#include <{}>\n", op.header(ctx).as_str())
});

shared_op!(AllocSharedOp, |op, ctx| {
    let name = op.get_result(ctx).name(ctx);
    let align = op.alignment(ctx).0;
    format!("extern __shared__ __align__({align}) char {name}[];")
});

/// Run on module
#[derive(Default)]
pub struct DeclareVectorTypesPass;

#[pass_name]
impl Pass for DeclareVectorTypesPass {
    fn run(
        &mut self,
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
                    if let Some(vector) = try_get_vector_type(ctx, val) {
                        vectors.insert((vector.inner.to_cpp(ctx), vector.vectorization), vector);
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
            let decl = DeclareVectorOp::new(ctx, vector.get_self_handle(ctx));
            decl.get_operation()
                .insert_at_front(module.get_body(ctx, 0), ctx);
            res.ir_changed |= IRStatus::Changed;
        }
        Ok(res)
    }
}

fn try_get_vector_type(ctx: &Context, value: Value) -> Option<VectorType> {
    let ty = value.get_type(ctx).deref(ctx);
    let elem = type_cast::<dyn HasElementType>(&*ty)?.element_type(ctx)?;
    elem.deref(ctx).downcast_ref().copied()
}

#[op_interface]
pub trait RequiresIncludesOp<T> {
    verify_op_succ!();
    fn includes(&self, ctx: &Context) -> Vec<String>;
}

#[type_interface]
pub trait RequiresIncludesType<T> {
    verify_ty_succ!();
    fn includes(&self, ctx: &Context) -> Vec<String>;
}

macro_rules! op_includes {
    ($target: ty, [$($ty: ty),*] => $inc: expr) => {
        $(#[pliron::derive::op_interface_impl]
        impl crate::shared::signature::RequiresIncludesOp<$target> for $ty {
            fn includes(&self, _ctx: &pliron::context::Context) -> Vec<String> {
                vec![$inc.into()]
            }
        })*
    };
}
pub(crate) use op_includes;

macro_rules! ty_includes {
    ($target: ty, [$($ty: ty),*] => $inc: expr) => {
        $(#[pliron::derive::type_interface_impl]
        impl crate::shared::signature::RequiresIncludesType<$target> for $ty {
            fn includes(&self, _ctx: &pliron::context::Context) -> Vec<String> {
                vec![$inc.into()]
            }
        })*
    };
}
pub(crate) use ty_includes;

macro_rules! nested_include_types {
    ($target: ty) => {
        #[type_interface_impl]
        impl RequiresIncludesType<$target> for PointerType {
            fn includes(&self, ctx: &Context) -> Vec<String> {
                let inner = self.inner.deref(ctx);
                if let Some(includes) = type_cast::<dyn RequiresIncludesType<$target>>(&*inner) {
                    includes.includes(ctx)
                } else {
                    vec![]
                }
            }
        }

        #[type_interface_impl]
        impl RequiresIncludesType<$target> for ArrayType {
            fn includes(&self, ctx: &Context) -> Vec<String> {
                let inner = self.inner.deref(ctx);
                if let Some(includes) = type_cast::<dyn RequiresIncludesType<$target>>(&*inner) {
                    includes.includes(ctx)
                } else {
                    vec![]
                }
            }
        }

        #[type_interface_impl]
        impl RequiresIncludesType<$target> for RuntimeArrayType {
            fn includes(&self, ctx: &Context) -> Vec<String> {
                let inner = self.inner.deref(ctx);
                if let Some(includes) = type_cast::<dyn RequiresIncludesType<$target>>(&*inner) {
                    includes.includes(ctx)
                } else {
                    vec![]
                }
            }
        }

        #[type_interface_impl]
        impl RequiresIncludesType<$target> for VectorType {
            fn includes(&self, ctx: &Context) -> Vec<String> {
                let inner = self.inner.deref(ctx);
                if let Some(includes) = type_cast::<dyn RequiresIncludesType<$target>>(&*inner) {
                    includes.includes(ctx)
                } else {
                    vec![]
                }
            }
        }
    };
}

nested_include_types!(Cuda);
nested_include_types!(Hip);
nested_include_types!(Metal);

/// Run on module
#[derive(Default)]
pub struct CollectIncludesPass<T: CppTarget> {
    _ty: PhantomData<T>,
}

#[pass_name]
impl<T: CppTarget> Pass for CollectIncludesPass<T> {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let module = op.as_op::<ModuleOp>(ctx).expect("Should be run on module");
        let mut includes = FxHashSet::default();

        walk_op(
            ctx,
            &mut includes,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, includes, node| {
                let mut ins = |val: Value| {
                    let ty = val.get_type(ctx).deref(ctx);
                    if let Some(includes_ty) = type_cast::<dyn RequiresIncludesType<T>>(&*ty) {
                        includes.extend(includes_ty.includes(ctx));
                    }
                };
                match node {
                    IRNode::Operation(op) => {
                        if let Some(res) = op.opt_result(ctx) {
                            ins(res);
                        }
                        let dyn_op = op.dyn_op(ctx);
                        if let Some(includes_op) = op_cast::<dyn RequiresIncludesOp<T>>(&*dyn_op) {
                            includes.extend(includes_op.includes(ctx));
                        }
                    }
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
        for include in includes {
            let decl = IncludeOp::new(ctx, include);
            decl.get_operation()
                .insert_at_front(module.get_body(ctx, 0), ctx);
            res.ir_changed |= IRStatus::Changed;
        }
        Ok(res)
    }
}

pub fn shared_memory_size(ctx: &Context, module: Ptr<Operation>) -> usize {
    let mut size = 0;
    visit_all_ops_of_type::<AllocSharedOp, _>(ctx, &mut size, module, |ctx, size, op| {
        *size += op.size(ctx).0;
    });
    size
}
