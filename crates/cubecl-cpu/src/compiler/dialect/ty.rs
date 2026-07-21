use super::prelude::*;
use super::to_llvm::CubeToLLVMType;
use cubecl_core::ir::types::scalar::{BoolType, Float32Type, Float64Type, IndexType};
use cubecl_core::ir::types::{PointerType as CubePointerType, VectorType as CubeVectorType};
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron_llvm::types::{
    PointerType as LlvmPointerType, VectorType as LlvmVectorType, VectorTypeKind,
};

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
impl_cube_to_llvm_type!(CubePointerType, self, ctx => LlvmPointerType::get(ctx, 0));
impl_cube_to_llvm_type!(CubeVectorType, self, ctx => LlvmVectorType::get(ctx, self.inner, self.vectorization as u32, VectorTypeKind::Fixed));
