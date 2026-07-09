use core::cell::Ref;

use cubecl_core::ir::{
    AddressSpace, ContextExt, GlobalState,
    attributes::{ATTR_BUFFER_BINDING, ATTR_READONLY, BufferBindingAttr, FuncInterface},
    interfaces::TypedExt,
    match_ty,
    prelude::*,
    types::{ArrayType, AtomicType, RuntimeArrayType, VectorType, scalar::*},
};
use pliron::{
    builtin::{attributes::TypeAttr, types::UnitType},
    r#type::TypedHandle,
};

use crate::{
    cuda::ty::*,
    error::CompileError,
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

impl<T: Type + ?Sized> TypeExt for T {}
pub trait TypeExt: Type {
    fn display(&self, ctx: &Context) -> String {
        format!("{}{}", self.get_type_id(), self.disp(ctx))
    }
}

macro_rules! is_one_of {
    ($ty: expr; $($types: ty),*) => {
        false $(|| $ty.is::<$types>())*
    };
}

pub trait TypedExtCPP: Typed {
    fn is_uniform_ptr(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.is::<UniformPointerType>()
    }

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

    fn is_fp8_fp6_fp4(&self, ctx: &Context) -> bool {
        self.is_float8(ctx) || self.is_float6(ctx) || self.is_float4(ctx)
    }

    fn is_packed_fp6_fp8_fp4(&self, ctx: &Context) -> bool {
        self.is_float8x2(ctx) || self.is_float6x2(ctx) || self.is_float4x2(ctx)
    }

    fn can_pack(&self, ctx: &Context) -> bool {
        if !self.is_vector(ctx) && !self.is_float4x2(ctx) {
            return false;
        }
        let scalar = self.scalar_ty(ctx);
        scalar.is_float16(ctx)
            || scalar.is_bfloat16(ctx)
            || scalar.is_fp8_fp6_fp4(ctx)
            || scalar.is_float4x2(ctx)
    }

    fn packed_type(&self, ctx: &Context) -> TypeHandle {
        assert!(self.can_pack(ctx), "Should be packable");
        // Already natively packed
        if self.is_float4x2(ctx) {
            return self.get_type(ctx);
        }
        let ty = self.get_type(ctx).deref(ctx);
        let vec = ty.downcast_ref::<VectorType>().unwrap();
        let scalar = vec.inner.deref(ctx);
        let scalar = match_ty!((scalar) {
            Float16Type => Float16x2Type::get(ctx).into(),
            BFloat16Type => BFloat16x2Type::get(ctx).into(),
            Float8E8M0Type => Float8E8M0x2Type::get(ctx).into(),
            Float8E5M2Type => Float8E5M2x2Type::get(ctx).into(),
            Float8E4M3Type => Float8E4M3x2Type::get(ctx).into(),
            Float6E2M3Type => Float6E2M3x2Type::get(ctx).into(),
            Float6E3M2Type => Float6E3M2x2Type::get(ctx).into(),
            Float4E2M1Type => Float4E2M1x2Type::get(ctx).into(),;
            _ => panic!("Unexpected type {}", scalar.display(ctx))
        });
        if vec.vectorization > 2 {
            VectorType::get(ctx, scalar, vec.vectorization / 2).to_handle()
        } else {
            scalar
        }
    }

    /// Whether the type is an integer that may be auto-promoted by C++
    /// They need special handling
    fn is_small_int(&self, ctx: &Context) -> bool {
        (self.is_int(ctx) || self.is_uint(ctx)) && self.scalar_ty(ctx).size(ctx) < 4
    }

    fn is_small_signed_int(&self, ctx: &Context) -> bool {
        self.is_int(ctx) && self.scalar_ty(ctx).size(ctx) < 4
    }

    fn is_small_unsigned_int(&self, ctx: &Context) -> bool {
        self.is_uint(ctx) && self.scalar_ty(ctx).size(ctx) < 4
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

pub fn ptr_constness(ctx: &Context, addr_space: AddressSpace) -> &'static str {
    match addr_space {
        AddressSpace::Global(idx) => match find_global_constness(ctx, idx) {
            true => "const",
            false => "",
        },
        AddressSpace::Shared | AddressSpace::Local => "",
    }
}

fn find_global_constness(ctx: &Context, idx: usize) -> bool {
    let func = ctx.aux_ty::<GlobalState>().entry_func;
    let num_args = func.get_entry_block(ctx).deref(ctx).get_num_arguments();
    let arg_pos = (0..num_args)
        .filter_map(|i| Some((i, func.get_arg_attr(ctx, i, &ATTR_BUFFER_BINDING)?)))
        .find(|(_, binding): &(_, Ref<'_, BufferBindingAttr>)| binding.buffer_pos == idx)
        .expect("Should exist");
    func.has_arg_attr(ctx, arg_pos.0, &ATTR_READONLY)
}

#[pliron_type(
    name = "cpp.info_ptr",
    format = "`<` $inner `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct UniformPointerType {
    pub inner: TypeHandle,
}

shared_ty!(UnitType, |_, _| "void".into());
shared_ty!(IntType, |ty, _| format!("int{}_t", ty.width));
shared_ty!(UIntType, |ty, _| format!("uint{}_t", ty.width));
shared_ty!(BoolType, |_, _| "bool".into());

shared_ty!(Float32Type, |_, _| "float".into());
shared_ty!(Float64Type, |_, _| "double".into());

shared_ty!(IndexType, |_, ctx| {
    ctx.address_type().unsigned_type().to_type(ctx).to_cpp(ctx)
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
