use super::prelude::*;
use cubecl_core::ir::types::{
    ArrayType, AtomicType,
    scalar::{
        Float8E4M3Type, Float8E5M2Type, Float8E8M0Type, Float16Type, Float32Type, Float64Type,
        FloatFlex32Type,
    },
};
use pliron::printable::Printable;

/// LLVM width of a `cube.index`, in bits.
///
/// Fixed at 64 **on purpose**, and not from the kernel's
/// [`AddressType`](cubecl_core::ir::AddressType) as one might expect — note
/// that `IndexType::size` does read `ctx.address_type()`, which defaults to
/// `U32`, so the two genuinely disagree and this is the one that decides the
/// emitted IR.
///
/// Narrowing it to follow the address type was tried and measured on gfx1151.
/// It is a pessimisation, because the pointer is 64-bit whatever the index is:
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
/// The 32-bit form still does the 64-bit shift and add to form the address, and
/// pays one more VALUE op to widen the index first. A decode benchmark measured
/// flat to slightly worse across three runs, and the memory probe did not move.
///
/// It is also **not correct as-is**: that widening is `v_ashrrev`, an
/// *arithmetic* shift, so the backend sign-extends the GEP index. The index
/// arithmetic itself is emitted unsigned (`udiv`, `icmp ult`), but an index at
/// or above 2^31 would address negatively — while `required_address_type` only
/// promotes to `U64` past `u32::MAX`. Narrowing would need an explicit `zext`
/// at the GEP, and would still not pay.
///
/// # The disagreement is not harmless, and is not fixed here
///
/// Keeping 64 is right for the *arithmetic*, but `IndexType::size` still
/// answers 4, and other code believes it. `shared_memory.rs` reserves a block
/// from `SizedType::size`, and `ArrayType::size` multiplies the inner one — so
/// a shared `[n x cube.index]` reserves `4n` bytes for a type this backend
/// emits as `[n x i64]` and writes `8n` into, overrunning whatever block was
/// laid out next. Nothing in tree allocates shared memory of index type today,
/// which is the only reason it has not bitten.
///
/// Fixing it properly means either sizing shared memory from the *converted*
/// LLVM type rather than the cube type, or making `IndexType::size` agree with
/// what this backend emits. Both are the owners' call; this comment exists so
/// the next reader does not conclude, as the old one invited, that two numbers
/// disagreeing is merely untidy.
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
