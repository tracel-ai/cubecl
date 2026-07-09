use cubecl_ir::{
    ContextExt,
    interfaces::TypedExt,
    types::{ArrayType, RuntimeArrayType, VectorType, scalar::*},
    verify_ty_succ,
};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    context::Context,
    derive::{type_interface, type_interface_impl},
    r#type::{TypeHandle, type_cast},
};
use pliron_spirv::types::{
    ArrayType as SpirvArrayType, FloatType, PointerType as SpirvPointerType,
    RuntimeArrayType as SpirvRuntimeArrayType, VectorType as SpirvVectorType,
};
use rspirv::spirv::FPEncoding;

pub fn ty_to_spirv_dialect(ctx: &Context, handle: TypeHandle) -> TypeHandle {
    let ty = handle.deref(ctx);
    if let Some(to_spirv) = type_cast::<dyn ToSpirvDialectType>(&*ty) {
        to_spirv.to_spirv_ty(ctx)
    } else {
        handle
    }
}

#[type_interface]
pub trait ToSpirvDialectType {
    verify_ty_succ!();
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle;
}

macro_rules! float_type {
    ($ty: ty, $width: literal, $encoding: expr) => {
        #[type_interface_impl]
        impl ToSpirvDialectType for $ty {
            fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
                FloatType::get(ctx, $width, $encoding).to_handle()
            }
        }
    };
}

float_type!(Float64Type, 64, None);
float_type!(Float32Type, 32, None);
float_type!(FloatFlex32Type, 32, None);
float_type!(Float16Type, 16, None);
float_type!(BFloat16Type, 16, Some(FPEncoding::BFloat16KHR));
float_type!(Float8E4M3Type, 8, Some(FPEncoding::Float8E4M3EXT));
float_type!(Float8E5M2Type, 8, Some(FPEncoding::Float8E5M2EXT));

#[type_interface_impl]
impl ToSpirvDialectType for BoolType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        IntegerType::get(ctx, 1, Signedness::Signless).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for IndexType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let address_ty = ctx.address_type();
        ty_to_spirv_dialect(ctx, address_ty.unsigned_type().to_type(ctx))
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for IntType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        IntegerType::get(ctx, self.width as u32, Signedness::Signless).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for UIntType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        IntegerType::get(ctx, self.width as u32, Signedness::Signless).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for VectorType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect(ctx, self.inner);
        SpirvVectorType::get(ctx, self.vectorization as u32, inner).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for ArrayType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect(ctx, self.inner);
        SpirvArrayType::get(ctx, self.length as u32, inner, None).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for RuntimeArrayType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect(ctx, self.inner);
        SpirvRuntimeArrayType::get(ctx, inner, Some(inner.size(ctx) as u32)).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for SpirvPointerType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect(ctx, self.element_type);
        SpirvPointerType::get(ctx, inner, self.storage_class).to_handle()
    }
}
