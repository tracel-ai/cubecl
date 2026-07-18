use cubecl_core::cmma::MatrixType;
use cubecl_ir::{
    AddressSpace, ContextExt,
    interfaces::TypedExt,
    prelude::FunctionTypeInterface,
    types::{
        ArrayType, AtomicType, MatrixIdent, MatrixScope, PointerType, RuntimeArrayType, VectorType,
        scalar::*,
        spirv::{ClampMode, TensorLayoutType, TensorViewType},
    },
    verify_ty_succ,
};
use pliron::{
    builtin::types::{FunctionType, IntegerType, Signedness},
    context::Context,
    derive::{type_interface, type_interface_impl},
    r#type::{TypeHandle, type_cast},
};
use pliron_spirv::types::{
    ArrayType as SpirvArrayType, FloatType, PointerType as SpirvPointerType,
    RuntimeArrayType as SpirvRuntimeArrayType, VectorType as SpirvVectorType,
    khr::CooperativeMatrixType, nv,
};
use rspirv::spirv::{
    CooperativeMatrixUse::{MatrixAKHR, MatrixAccumulatorKHR, MatrixBKHR},
    FPEncoding, Scope, StorageClass, TensorClampMode,
};

pub fn ty_to_spirv_dialect(ctx: &Context, handle: impl Into<TypeHandle>) -> TypeHandle {
    let handle = handle.into();
    let ty = handle.deref(ctx);
    if let Some(to_spirv) = type_cast::<dyn ToSpirvDialectType>(&*ty) {
        to_spirv.to_spirv_ty(ctx)
    } else {
        handle
    }
}

pub fn ty_to_spirv_dialect_explicit_layout(
    ctx: &Context,
    handle: impl Into<TypeHandle>,
) -> TypeHandle {
    let handle = handle.into();
    let ty = handle.deref(ctx);
    if let Some(to_spirv) = type_cast::<dyn ToSpirvDialectType>(&*ty) {
        to_spirv.to_spirv_ty_explicit_layout(ctx)
    } else {
        handle
    }
}

#[type_interface]
pub trait ToSpirvDialectType {
    verify_ty_succ!();
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle;
    fn to_spirv_ty_explicit_layout(&self, ctx: &Context) -> TypeHandle {
        self.to_spirv_ty(ctx)
    }
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

/// Erase sign to simplify SPIR-V, it's ignored anyways
#[type_interface_impl]
impl ToSpirvDialectType for IntegerType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        IntegerType::get(ctx, self.width(), Signedness::Signless).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for AtomicType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        ty_to_spirv_dialect(ctx, self.inner)
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

    fn to_spirv_ty_explicit_layout(&self, ctx: &Context) -> TypeHandle {
        let stride = self.inner.size(ctx) as u32;
        let inner = ty_to_spirv_dialect(ctx, self.inner);
        SpirvArrayType::get(ctx, self.length as u32, inner, Some(stride)).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for RuntimeArrayType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect(ctx, self.inner);
        SpirvRuntimeArrayType::get(ctx, inner, None).to_handle()
    }

    fn to_spirv_ty_explicit_layout(&self, ctx: &Context) -> TypeHandle {
        let stride = self.inner.size(ctx) as u32;
        let inner = ty_to_spirv_dialect(ctx, self.inner);
        SpirvRuntimeArrayType::get(ctx, inner, Some(stride)).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for PointerType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect(ctx, self.inner);
        let storage_class = match self.address_space {
            AddressSpace::Global(_) => StorageClass::PhysicalStorageBuffer,
            AddressSpace::Shared => StorageClass::Workgroup,
            AddressSpace::Local => StorageClass::Function,
        };
        SpirvPointerType::get(ctx, inner, storage_class).to_handle()
    }

    fn to_spirv_ty_explicit_layout(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect_explicit_layout(ctx, self.inner);
        let storage_class = match self.address_space {
            AddressSpace::Global(_) => StorageClass::PhysicalStorageBuffer,
            AddressSpace::Shared => StorageClass::Workgroup,
            AddressSpace::Local => StorageClass::Function,
        };
        SpirvPointerType::get(ctx, inner, storage_class).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for SpirvPointerType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect(ctx, self.element_type);
        SpirvPointerType::get(ctx, inner, self.storage_class).to_handle()
    }

    fn to_spirv_ty_explicit_layout(&self, ctx: &Context) -> TypeHandle {
        let inner = ty_to_spirv_dialect_explicit_layout(ctx, self.element_type);
        SpirvPointerType::get(ctx, inner, self.storage_class).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for MatrixType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let component = ty_to_spirv_dialect(ctx, self.elem_ty);
        let scope = match self.scope {
            MatrixScope::Plane => Scope::Subgroup,
            MatrixScope::Cube => Scope::Workgroup,
        };
        let (rows, cols, use_) = match self.ident {
            MatrixIdent::A => (self.shape.m, self.shape.k, MatrixAKHR),
            MatrixIdent::B => (self.shape.k, self.shape.n, MatrixBKHR),
            MatrixIdent::Accumulator => (self.shape.m, self.shape.n, MatrixAccumulatorKHR),
        };
        CooperativeMatrixType::get(ctx, component, scope, rows as u32, cols as u32, use_).into()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for FunctionType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let args = self.arg_types().into_iter();
        let args = args.map(|ty| ty_to_spirv_dialect(ctx, ty)).collect();
        let res = self.res_types().into_iter();
        let res = res.map(|ty| ty_to_spirv_dialect(ctx, ty)).collect();
        FunctionType::get(ctx, args, res).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for TensorLayoutType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let clamp_mode = match self.clamp_mode {
            ClampMode::Undefined => TensorClampMode::Undefined,
            ClampMode::Constant(_) => TensorClampMode::Constant,
            ClampMode::ClampToEdge => TensorClampMode::ClampToEdge,
            ClampMode::Repeat => TensorClampMode::Repeat,
            ClampMode::RepeatMirrored => TensorClampMode::RepeatMirrored,
        };
        nv::TensorLayoutType::get(ctx, self.rank as u32, clamp_mode).to_handle()
    }
}

#[type_interface_impl]
impl ToSpirvDialectType for TensorViewType {
    fn to_spirv_ty(&self, ctx: &Context) -> TypeHandle {
        let permutation = self.permutation.iter().map(|it| *it as u32).collect();
        nv::TensorViewType::get(ctx, self.rank as u32, self.has_dims, permutation).to_handle()
    }
}
