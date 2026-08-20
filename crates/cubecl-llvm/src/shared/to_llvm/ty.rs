use super::prelude::*;
use cubecl_core::ir::types::{
    ArrayType, AtomicType,
    scalar::{
        Float8E4M3Type, Float8E5M2Type, Float16Type, Float32Type, Float64Type, FloatFlex32Type,
    },
};
use pliron::printable::Printable;

/// LLVM width of a `cube.index`. `IndexType` is `size_of::<u64>()`, so it maps to `i64`.
pub const INDEX_WIDTH: u32 = 64;

macro_rules! impl_cube_to_llvm_type {
    ($src:ty, $self:ident, $ctx:ident => $body:expr) => {
        #[type_interface_impl]
        impl CubeToLLVMType for $src {
            fn convert(&$self, $ctx: &Context) -> TypeHandle {
                ($body).into()
            }
        }
    };
}

impl_cube_to_llvm_type!(IntegerType, self, ctx => IntegerType::get(ctx, self.width(), Signedness::Signless));
impl_cube_to_llvm_type!(BoolType, self, ctx => IntegerType::get(ctx, 1, Signedness::Signless));
impl_cube_to_llvm_type!(IndexType, self, ctx => IntegerType::get(ctx, INDEX_WIDTH, Signedness::Signless));
impl_cube_to_llvm_type!(Float64Type, self, ctx => FP64Type::get(ctx));
impl_cube_to_llvm_type!(Float32Type, self, ctx => FP32Type::get(ctx));
impl_cube_to_llvm_type!(FloatFlex32Type, self, ctx => FP32Type::get(ctx));
impl_cube_to_llvm_type!(Float16Type, self, ctx => FP16Type::get(ctx));
impl_cube_to_llvm_type!(Float8E4M3Type, self, ctx => IntegerType::get(ctx, 8, Signedness::Signless));
impl_cube_to_llvm_type!(Float8E5M2Type, self, ctx => IntegerType::get(ctx, 8, Signedness::Signless));
impl_cube_to_llvm_type!(CubePointerType, self, ctx => LlvmPointerType::get(ctx, 0));
impl_cube_to_llvm_type!(CubeVectorType, self, ctx => LlvmVectorType::get(ctx, cube_type_to_llvm(ctx, self.inner), self.vectorization as u32, VectorTypeKind::Fixed));
impl_cube_to_llvm_type!(AtomicType, self, ctx => cube_type_to_llvm(ctx, self.inner));
impl_cube_to_llvm_type!(ArrayType, self, ctx => {
    let inner = cube_type_to_llvm(ctx, self.inner);
    LlvmArrayType::get(ctx, inner, self.length as u64)
});

/// Convert a cubecl type to its LLVM-dialect equivalent, or return it unchanged when no
/// conversion applies.
pub fn cube_type_to_llvm(ctx: &Context, ty: TypeHandle) -> TypeHandle {
    type_cast::<dyn CubeToLLVMType>(&*ty.deref(ctx))
        .map(|convertible| convertible.convert(ctx))
        .unwrap_or(ty)
}

pub fn scalar_alignment(ctx: &Context, ty: TypeHandle) -> u32 {
    let scalar = {
        let ty = ty.deref(ctx);
        type_cast::<dyn ScalarizableType>(&*ty).map(|s| s.scalar_type(ctx))
    }
    .unwrap_or(ty);

    let scalar = scalar.deref(ctx);
    let scalar = type_cast::<dyn AlignedType>(&*scalar);
    if scalar.is_none() {
        println!("{}", ty.disp(ctx));
    }
    scalar
        .expect("load/store value type must implement AlignedType")
        .align(ctx) as u32
}

#[type_interface]
pub trait LlvmTypeToMangledOverload {
    verify_ty_succ!();
    fn to_string(&self, ctx: &Context) -> String;
}

macro_rules! impl_llvm_type_to_mangled_overload {
    ($src:ty, $self:ident, $ctx:ident => $body:expr) => {
        #[type_interface_impl]
        impl LlvmTypeToMangledOverload for $src {
            fn to_string(&$self, $ctx: &Context) -> String {
                $body
            }
        }
    };
}

impl_llvm_type_to_mangled_overload!(IntegerType, self, _ctx => format!("i{}", self.width()));
impl_llvm_type_to_mangled_overload!(FP16Type, self, _ctx => "f16".to_string());
impl_llvm_type_to_mangled_overload!(FP32Type, self, _ctx => "f32".to_string());
impl_llvm_type_to_mangled_overload!(FP64Type, self, _ctx => "f64".to_string());
impl_llvm_type_to_mangled_overload!(LlvmVectorType, self, ctx => {
    let prefix = if self.is_scalable() {
        "nx"
    } else {
        ""
    };
    let (n, elem) = (self.num_elements(), self.elem_type());
    format!("{prefix}v{n}{}", llvm_mangled_ty(ctx, elem))
});

/// Convert a llvm type to the string
pub fn llvm_mangled_ty(ctx: &Context, ty: TypeHandle) -> String {
    type_cast::<dyn LlvmTypeToMangledOverload>(&*ty.deref(ctx))
        .map(|ty| ty.to_string(ctx))
        .expect("Type not supported for overloading of intrinsic")
}
