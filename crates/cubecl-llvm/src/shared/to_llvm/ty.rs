use super::prelude::*;
use cubecl_core::ir::types::{
    ArrayType, AtomicType,
    scalar::{
        Float8E4M3Type, Float8E5M2Type, Float8E8M0Type, Float16Type, Float32Type, Float64Type,
        FloatFlex32Type,
    },
};
use cubecl_core::ir::{AddressType, ContextExt};
use pliron::printable::Printable;

use crate::target::{CtxTarget, LlvmTarget};

/// LLVM width of a `cube.index`, in bits.
///
/// # AMDGPU and the CPU compute at 64 bits, whatever the address type says
///
/// Note that `IndexType::size` does read [`ContextExt::address_type`], which
/// defaults to `U32`, so on those targets the two genuinely disagree and this
/// is the one that decides the emitted IR.
///
/// Narrowing them to follow the address type was tried and measured on
/// gfx1151. It is a pessimisation there, because the pointer is 64-bit
/// whatever the index is:
///
/// ```text
/// index i64                           index i32
/// ---------                           ---------
/// v_lshlrev_b64   v[10:11], 4, ...    v_ashrrev_i32 v9, 31, v8   <- extra
/// v_add_co_u32    v10, s0, v10        v_lshlrev_b64 v[8:9], 4, ...
/// v_add_co_ci_u32 v11, s1, v11        v_add_co_u32  v8, s0, v8
/// global_load_b128                    v_add_co_ci_u32 v9, s1, v9
///                                     global_load_b128
/// ```
///
/// The 32-bit form still does the 64-bit shift and add to form the address,
/// and pays one more VALU op to widen the index first. A decode benchmark
/// measured flat to slightly worse across three runs, and the memory probe did
/// not move.
///
/// The disagreement is not harmless on those targets: `shared_memory.rs`
/// reserves a block from `SizedType::size`, and `ArrayType::size` multiplies
/// the inner one — so a shared `[n x cube.index]` reserves `4n` bytes for a
/// type this backend emits as `[n x i64]` and writes `8n` into, overrunning
/// whatever block was laid out next. Nothing in tree allocates shared memory
/// of index type today, which is the only reason it has not bitten.
///
/// # NVPTX follows the address type
///
/// The same measurement comes out the other way on NVIDIA, because the widening
/// AMD pays a VALU op for is free here: `mad.wide.u32` multiplies two 32-bit
/// operands into a 64-bit result, so a `zext`ed index folds into the address
/// arithmetic that had to happen anyway. What it saves is everything upstream
/// of that. 64-bit integer arithmetic is emulated on NVIDIA — `mul.lo.s64` is
/// several 32-bit multiplies, a variable `shr.u64` several shifts — and a
/// tiled matmul's addressing is mostly multiplies, shifts and bounds compares.
/// Measured on a cmma GEMM, following the address type took the emitted PTX
/// from 458 64-bit operations to 95 — fewer than the C++ backend's 206, which
/// keeps the same arithmetic in `unsigned` and widens once at the pointer.
///
/// Following the address type rather than pinning 32 keeps the buffers past
/// `u32::MAX` elements that `AddressType::from_len` promotes for, and makes
/// `IndexType::size` agree with what is emitted, so the shared-memory hazard
/// above does not arise here.
///
/// [`ContextExt::address_type`]: cubecl_core::ir::ContextExt::address_type
pub fn index_width(ctx: &Context) -> u32 {
    match ctx.target() {
        LlvmTarget::Nvptx => match ctx.address_type() {
            AddressType::U32 => 32,
            AddressType::U64 => 64,
        },
        LlvmTarget::AmdGpu | LlvmTarget::Cpu => 64,
    }
}

/// The width a `getelementptr` index is taken at, which is the pointer's.
///
/// A narrower index has to be widened explicitly, and *zero*-extended: `cube.index` is
/// unsigned, where the `getelementptr` operand is signed and would otherwise address
/// negatively from 2^31 up.
pub const GEP_INDEX_WIDTH: u32 = 64;

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
impl_cube_to_llvm_type!(IndexType, self, ctx => IntegerType::get(ctx, index_width(ctx), Signedness::Signless));
impl_cube_to_llvm_type!(Float64Type, self, ctx => FP64Type::get(ctx));
impl_cube_to_llvm_type!(Float32Type, self, ctx => FP32Type::get(ctx));
impl_cube_to_llvm_type!(FloatFlex32Type, self, ctx => FP32Type::get(ctx));
impl_cube_to_llvm_type!(Float16Type, self, ctx => FP16Type::get(ctx));
impl_cube_to_llvm_type!(Float8E4M3Type, self, ctx => IntegerType::get(ctx, 8, Signedness::Signless));
impl_cube_to_llvm_type!(Float8E5M2Type, self, ctx => IntegerType::get(ctx, 8, Signedness::Signless));
impl_cube_to_llvm_type!(Float8E8M0Type, self, ctx => IntegerType::get(ctx, 8, Signedness::Signless));
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

/// The alignment of `ty` itself, which for a vector is the whole vector's.
///
/// What an ordinary load or store of a vectorized value promises: `CubeCL` vectorizes an access
/// only where the layout lets it, so a `Vector<f16, 8>` read out of a buffer of them sits on a
/// 16 byte boundary, the same promise the C++ backends make by giving the type an
/// `alignas(16)`. Saying less than that is not merely conservative -- a target legalizes an
/// under-aligned vector access into one scalar access per element, so a 128 bit store becomes
/// eight 16 bit ones, which is most of the memory traffic of a kernel that moves tiles around.
///
/// Not for an access whose address is arithmetic the frontend did not vectorize: a matrix tile
/// is `stride` elements per row and a caller may pad that stride, so those keep
/// [`scalar_alignment`].
pub fn type_alignment(ctx: &Context, ty: TypeHandle) -> u32 {
    let ty = ty.deref(ctx);
    type_cast::<dyn AlignedType>(&*ty)
        .expect("load/store value type must implement AlignedType")
        .align(ctx) as u32
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
// A pointer mangles as its address space, which is how the intrinsics overloaded on one --
// the NVPTX matrix loads and stores -- tell `p0` from `p3`.
impl_llvm_type_to_mangled_overload!(LlvmPointerType, self, _ctx => format!("p{}", self.address_space()));
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
