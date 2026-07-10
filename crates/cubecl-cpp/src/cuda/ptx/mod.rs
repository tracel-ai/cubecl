use cubecl_core::{
    self as cubecl,
    frontend::barrier::Barrier,
    intrinsic,
    ir::{
        AddressSpace,
        interfaces::TypeExt,
        prelude::*,
        types::{
            PointerType,
            barrier::{BarrierLevel, BarrierType},
            cuda::TensorMapType,
        },
    },
    prelude::*,
};

mod asm;
mod mma;

pub use asm::*;
pub use mma::*;
use pliron::{
    builtin::types::{IntegerType, Signedness},
    printable::Printable,
    verify_err,
};

pub mod copy_async;
pub mod tma_load_im2col;

use crate::{cuda::cuda_op_with_out, shared::ty::TypeExtCPP};

/// Equivalent of `__cvta_generic_to_shared`, required when a PTX instruction uses a specific
/// `.shared` modifier. It should only be used to cast a pointer for use in that specific context,
/// and using it without adding the `.shared` modifier will break. Using shared addresses in generic
/// instructions will also break, which is why this isn't automatically applied in `InlinePtxOp`.
#[cube_op(name = "cuda.generic_to_shared", format = "$0 ` : ` type($0)")]
#[result_ty(fixed = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle())]
pub struct GenericToSharedOp {
    ptr: Value,
}

cuda_op_with_out!(GenericToSharedOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    format!("__cvta_generic_to_shared({ptr})")
});

#[cube]
pub fn generic_to_shared<T: CubePrimitive>(ptr: *const T) -> u32 {
    intrinsic!(|scope| {
        let cvt = GenericToSharedOp::new(scope.ctx_mut(), unsafe { *ptr }.value(scope));
        scope.register_with_result(&cvt).into()
    })
}

/// Returns the shared address directly because it's always combined with `generic_to_shared` in
/// practice, and it makes the types easier to deal with.
#[cube_op(
    name = "cuda.barrier_native_handle",
    format = "$0 ` : ` type($0)",
    verifier = "custom"
)]
#[result_ty(fixed = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle())]
#[op_interfaces(OperandNOfType<0, PointerType>)]
pub struct BarrierNativeHandleOp {
    bar_ptr: Value,
}

impl Verify for BarrierNativeHandleOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let barrier_ty = self.bar_ptr(ctx).get_type(ctx).as_ptr(ctx);

        if !barrier_ty.inner.deref(ctx).is::<BarrierType>() {
            let expected = PointerType::get(
                ctx,
                BarrierType::get(ctx, BarrierLevel::Cube).into(),
                AddressSpace::Shared,
            )
            .to_handle();
            return verify_err!(
                loc,
                OperandNOfTypeError::AllOperandsOfTypeVerifyErr(
                    expected.disp(ctx).to_string(),
                    self.bar_ptr(ctx).get_type(ctx).disp(ctx).to_string()
                )
            )?;
        }
        Ok(())
    }
}

cuda_op_with_out!(BarrierNativeHandleOp, |op, ctx| {
    let ptr = op.bar_ptr(ctx).name(ctx);
    format!("__cvta_generic_to_shared(cuda::device::barrier_native_handle(*{ptr}))")
});

#[cube]
pub fn barrier_native_handle(bar: &Barrier) -> u32 {
    intrinsic!(|scope| {
        let handle = BarrierNativeHandleOp::new(scope.ctx_mut(), bar.value(scope));
        scope.register_with_result(&handle).into()
    })
}

/// Tensor maps are weird because they're passed as a reference but consumed by pointer. Don't want
/// to add a generic `ReferenceOp`, so for now just do a tensor-map specific reference. That way
/// it doesn't need to deal with Metal address space nonsense. Returns the raw address as `u64`
/// since it's always used for PTX anyways, and that coerces pointers to u64.
#[cube_op(name = "cuda.tensor_map_addr", format = "$0 ` : ` type($0)")]
#[result_ty(fixed = IntegerType::get(ctx, 64, Signedness::Unsigned).to_handle())]
#[op_interfaces(OperandNOfType<0, TensorMapType>)]
pub struct TensorMapAddrOp {
    tensor_map: Value,
}

cuda_op_with_out!(TensorMapAddrOp, |op, ctx| {
    let tensor_map = op.tensor_map(ctx).name(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("reinterpret_cast<{out_ty}>(&{tensor_map})")
});

#[cube]
pub fn tensor_map_address<T: CubePrimitive, K: TensorMapKind>(tensor_map: &TensorMap<T, K>) -> u64 {
    intrinsic!(|scope| {
        let addr = TensorMapAddrOp::new(scope.ctx_mut(), tensor_map.value(scope));
        scope.register_with_result(&addr).into()
    })
}
