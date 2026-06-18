use core::cell::Ref;

use cubecl_core::ir::{
    ContextExt,
    attributes::FuncInterface,
    dialect::{
        OperationPtrExt,
        general::CopyOp,
        memory::{DeclareVariableOp, IndexOp},
    },
    interfaces::TypedExt,
    prelude::*,
    types::{
        ArrayType, AtomicType, PointerType as CubePointerType, RuntimeArrayType, VectorType,
        scalar::*,
    },
};
use cubecl_runtime::kernel::Visibility;
use pliron::{
    builtin::{attributes::TypeAttr, ops::FuncOp, types::UnitType},
    graph::walkers::uninterruptible::immutable::walk_op,
    r#type::TypedHandle,
};

use crate::{
    cuda::ty::*,
    error::CompileError,
    shared::{ATTR_CONST, CompilationState},
    target::{Shared, dispatch_target},
};

macro_rules! shared_ty {
    ($ty: ty, $impl: expr) => {
        #[type_interface_impl]
        impl TypeToCPP<Shared> for $ty {
            fn to_cpp(&self, ctx: &Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}

pub trait TypeExtCPP {
    fn to_cpp(&self, ctx: &Context) -> String;
}
impl TypeExtCPP for Ref<'_, dyn Type> {
    fn to_cpp(&self, ctx: &Context) -> String {
        let target_cpp = dispatch_target!(ctx, {
            let inner_cpp = type_cast::<dyn TypeToCPP<Target>>(&**self);
            inner_cpp.map(|it| it.to_cpp(ctx))
        });
        if let Some(cpp) = target_cpp {
            return cpp;
        }
        let shared = type_cast::<dyn TypeToCPP<Shared>>(&**self)
            .ok_or_else(|| {
                CompileError::UnsupportedType(format!("{}{}", self.get_type_id(), self.disp(ctx)))
            })
            .unwrap(); // Unwrap for now because handling errors everywhere is annoying
        shared.to_cpp(ctx)
    }
}
impl TypeExtCPP for TypeHandle {
    fn to_cpp(&self, ctx: &Context) -> String {
        self.deref(ctx).to_cpp(ctx)
    }
}
impl<T: Type> TypeExtCPP for TypedHandle<T> {
    fn to_cpp(&self, ctx: &Context) -> String {
        self.to_handle().to_cpp(ctx)
    }
}
impl TypeExtCPP for TypeAttr {
    fn to_cpp(&self, ctx: &Context) -> String {
        self.get_type(ctx).to_cpp(ctx)
    }
}

macro_rules! is_one_of {
    ($ty: expr; $($types: ty),*) => {
        true $(&& $ty.is::<$types>())*
    };
}

pub trait TypedExtCPP: Typed {
    fn is_half(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        is_one_of!(ty; Float16Type, BFloat16Type)
    }

    fn is_half2(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        is_one_of!(ty; Float16x2Type, BFloat16x2Type)
    }

    fn is_float8(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        is_one_of!(ty; Float8E8M0Type, Float8E5M2Type, Float8E4M3Type)
    }

    fn is_float8x2(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        is_one_of!(ty; Float8E8M0x2Type, Float8E5M2x2Type, Float8E4M3x2Type)
    }

    fn is_float6(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        is_one_of!(ty; Float6E3M2Type, Float6E2M3Type)
    }

    fn is_float6x2(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        is_one_of!(ty; Float6E3M2x2Type, Float6E2M3x2Type)
    }

    fn is_float4(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.is::<Float4E2M1Type>()
    }

    fn is_float4x2(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.is::<Float4E2M1x2Type>()
    }

    fn is_fp6_fp8_fp4(&self, ctx: &Context) -> bool {
        self.is_float8(ctx) || self.is_float6(ctx) || self.is_float4(ctx)
    }

    fn is_packed_fp6_fp8_fp4(&self, ctx: &Context) -> bool {
        self.is_float8x2(ctx) || self.is_float6x2(ctx) || self.is_float4x2(ctx)
    }

    /// Whether the type is an integer that may be auto-promoted by C++
    /// They need special handling
    fn is_small_int(&self, ctx: &Context) -> bool {
        (self.is_int(ctx) || self.is_uint(ctx)) && self.scalar_ty(ctx).size(ctx) < 4
    }

    fn unpacked_size_bits(&self, ctx: &Context) -> usize {
        self.size(ctx) / self.packing_factor(ctx)
    }

    fn address_space_cpp(&self, ctx: &Context) -> AddressSpace {
        let ty = self.get_type(ctx).deref(ctx);
        let ptr = ty.downcast_ref::<PointerType>().expect("Should be ptr");
        ptr.address_space
    }
}
impl<T: Typed> TypedExtCPP for T {}

#[type_interface]
pub trait TypeToCPP<T> {
    verify_ty_succ!();
    fn to_cpp(&self, ctx: &Context) -> String;
}

shared_ty!(VectorType, |ty, ctx| {
    format!("{}_{}", ty.inner.to_cpp(ctx), ty.vectorization)
});

shared_ty!(AtomicType, |ty, ctx| ty.inner.to_cpp(ctx));
shared_ty!(RuntimeArrayType, |ty, ctx| ty.inner.to_cpp(ctx));

shared_ty!(ArrayType, |ty, ctx| {
    format!("array<{}, {}>", ty.inner.to_cpp(ctx), ty.length)
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[format]
pub enum AddressSpace {
    Global(Visibility),
    Shared,
    Local,
}

impl AddressSpace {
    pub fn display_cuda_hip(&self) -> String {
        match self {
            AddressSpace::Global(Visibility::Read | Visibility::Uniform) => "const ".into(),
            _ => "".into(),
        }
    }
}

#[pliron_type(
    name = "cpp.ptr",
    format = "`<` $inner `, ` $address_space `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct PointerType {
    pub inner: TypeHandle,
    pub address_space: AddressSpace,
}

shared_ty!(UnitType, |_, _| "void".into());
shared_ty!(IntType, |ty, _| format!("int{}_t", ty.width));
shared_ty!(UIntType, |ty, _| format!("uint{}_t", ty.width));
shared_ty!(BoolType, |_, _| "bool".into());

shared_ty!(Float32Type, |_, _| "float".into());
shared_ty!(Float64Type, |_, _| "double".into());

shared_ty!(IndexType, |_, ctx| {
    ctx.aux_ty::<CompilationState>().address_type.to_cpp(ctx)
});

/// Vector of three unsigned integers. This is the only native vector type that's actually revelant
/// for codegen, so we can just special case it and only use it where necessary (builtin types).
#[pliron_type(
    name = "cpp.uvec3",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct Uvec3Type;

shared_ty!(Uvec3Type, |_, _| "uint3".into());

#[pliron_type(
    name = "cpp.info_st",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct InfoStructType;

shared_ty!(InfoStructType, |_, _| "info_st".into());

#[op_interface]
trait PtrDefiningOp: OneResultInterface {
    verify_op_succ!();
    fn remap_type(&self, ctx: &Context) -> TypeHandle;
}

#[op_interface_impl]
impl PtrDefiningOp for DeclareVariableOp {
    fn remap_type(&self, ctx: &Context) -> TypeHandle {
        let addr_space = match self.addr_space(ctx).0 {
            cubecl_core::ir::AddressSpace::Global(_) => unreachable!(),
            cubecl_core::ir::AddressSpace::Shared => AddressSpace::Shared,
            cubecl_core::ir::AddressSpace::Local => AddressSpace::Local,
        };
        let value_ty = self.value_ty(ctx).get_type(ctx);
        PointerType::get(ctx, value_ty, addr_space).to_handle()
    }
}

#[op_interface_impl]
impl PtrDefiningOp for IndexOp {
    fn remap_type(&self, ctx: &Context) -> TypeHandle {
        let source_ty = self.base(ctx).get_type(ctx).deref(ctx);
        let source_ty = source_ty.downcast_ref::<PointerType>().unwrap();
        let dest_ty = self.get_result(ctx).get_type(ctx).deref(ctx);
        let dest_ty = dest_ty.downcast_ref::<CubePointerType>().unwrap();
        PointerType::get(ctx, dest_ty.inner, source_ty.address_space).to_handle()
    }
}

#[op_interface_impl]
impl PtrDefiningOp for CopyOp {
    fn remap_type(&self, ctx: &Context) -> TypeHandle {
        self.value(ctx).get_type(ctx)
    }
}

#[derive(Default)]
pub struct ConvertPtrPass;

impl Pass for ConvertPtrPass {
    fn name(&self) -> &str {
        "convert_ptr_args"
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let func = op.as_op::<FuncOp>(ctx).unwrap();
        let args = func
            .get_entry_block(ctx)
            .deref(ctx)
            .arguments()
            .collect::<Vec<_>>();
        for (i, arg) in args.into_iter().enumerate() {
            let is_const = func.get_arg_attr(ctx, i, &ATTR_CONST).is_some();
            if let Some(&ptr) = arg
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<CubePointerType>()
            {
                let vis = match is_const {
                    true => Visibility::Read,
                    false => Visibility::ReadWrite,
                };
                let new_ty = PointerType::get(ctx, ptr.inner, AddressSpace::Global(vis));
                arg.set_type(ctx, new_ty.into());
                res.ir_changed |= IRStatus::Changed;
            }
        }

        walk_op(
            ctx,
            &mut res,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, res, node| {
                if let IRNode::Operation(op) = node
                    && let Some(defines_ptr) = op_cast::<dyn PtrDefiningOp>(&*op.dyn_op(ctx))
                {
                    let new_ty = defines_ptr.remap_type(ctx);
                    defines_ptr.get_result(ctx).set_type(ctx, new_ty);
                    res.ir_changed |= IRStatus::Changed;
                }
            },
        );

        Ok(res)
    }
}
