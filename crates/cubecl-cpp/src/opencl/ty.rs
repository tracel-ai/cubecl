use cubecl_core::ir::{
    AddressSpace, ContextExt,
    prelude::*,
    types::{
        ArrayType, AtomicType, PointerType, VectorType,
        scalar::{
            Float4E2M1x2Type, Float8E4M3Type, Float8E5M2Type, Float8E8M0Type, Float16Type,
            Float32Type, Float64Type, FloatFlex32Type, IndexType,
        },
    },
};
use pliron::builtin::types::{IntegerType, Signedness};

use crate::{
    error::{CompileError, EmissionErrors},
    shared::ty::{InfoStructType, TypeExtCPP, TypeToCPP, UniformPointerType, ptr_constness},
    target::OpenCl,
};

macro_rules! opencl_ty {
    ($ty: ty, $impl: expr) => {
        #[type_interface_impl]
        impl TypeToCPP<OpenCl> for $ty {
            fn to_cpp(&self, ctx: &Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use opencl_ty;

opencl_ty!(Float16Type, |_, _| "half".into());
opencl_ty!(Float8E4M3Type, |_, _| "uint8_t".into());
opencl_ty!(Float8E5M2Type, |_, _| "uint8_t".into());
opencl_ty!(Float8E8M0Type, |_, _| "uint8_t".into());
opencl_ty!(Float4E2M1x2Type, |_, _| "uint8_t".into());

// Atomics take non-const pointers, even into a buffer the kernel only reads.
opencl_ty!(PointerType, |ty, ctx| format!(
    "{} {} {}*",
    ptr_space(ty.address_space),
    ty.inner.to_cpp(ctx),
    match ty.inner.deref(ctx).is::<AtomicType>() {
        true => "",
        false => ptr_constness(ctx, ty.address_space),
    },
));
opencl_ty!(UniformPointerType, |ty, ctx| format!(
    "__constant {}*",
    ty.inner.to_cpp(ctx)
));

/// The `OpenCL` C address space qualifier of `addr_space`.
pub fn ptr_space(addr_space: AddressSpace) -> &'static str {
    match addr_space {
        AddressSpace::Global(_) => "__global",
        AddressSpace::Shared => "__local",
        AddressSpace::Local => "__private",
    }
}

// C11 atomics of OpenCL C 3.0: one named type per value type.
opencl_ty!(AtomicType, |ty, ctx| {
    let inner = ty.inner.deref(ctx);
    if let Some(int) = inner.downcast_ref::<IntegerType>() {
        let signed = int.signedness() == Signedness::Signed;
        match (int.width(), signed) {
            (32, true) => "atomic_int",
            (32, false) => "atomic_uint",
            (64, true) => "atomic_long",
            (64, false) => "atomic_ulong",
            (width, _) => {
                let error = CompileError::UnsupportedType(format!("atomic {width}-bit integers"));
                ctx.aux_ty::<EmissionErrors>().record(error);
                "atomic_int"
            }
        }
        .into()
    } else {
        format!("atomic_{}", ty.inner.to_cpp(ctx))
    }
});

// Arrays are values in the IR (copied, returned): a struct around a C array, declared once per
// element type and length by the module (`signature::declare_arrays`).
opencl_ty!(ArrayType, |ty, ctx| array_name(ctx, ty));

/// The name of the struct the module declares for the array type `ty`.
pub fn array_name(ctx: &Context, ty: &ArrayType) -> String {
    let inner = ty.inner.to_cpp(ctx).replace(' ', "_");
    format!("array_{inner}_{}", ty.length)
}

// C names a struct with its keyword.
opencl_ty!(InfoStructType, |_, _| "struct info_st".into());

/// `OpenCL`'s own vector type for `vector`, where it has one (2, 4, 8 or 16 numbers; no bool, and no 3: `OpenCL` pads it
/// to 4); otherwise the shared struct.
pub fn native_vector(ctx: &Context, vector: &VectorType) -> Option<String> {
    if !matches!(vector.vectorization, 2 | 4 | 8 | 16) {
        return None;
    }
    Some(format!(
        "{}{}",
        scalar_name(ctx, vector.inner)?,
        vector.vectorization
    ))
}

/// `OpenCL`'s built-in name of a numeric scalar type (`int`, `uchar`, `half`...): `usize` as the address type's
/// unsigned integer.
pub fn scalar_name(ctx: &Context, ty: TypeHandle) -> Option<&'static str> {
    let ty = match ty.deref(ctx).is::<IndexType>() {
        true => ctx.address_type().unsigned_type().to_type(ctx),
        false => ty,
    };
    let inner = ty.deref(ctx);
    let scalar = if let Some(int) = inner.downcast_ref::<IntegerType>() {
        let (signed, unsigned) = match int.width() {
            8 => ("char", "uchar"),
            16 => ("short", "ushort"),
            32 => ("int", "uint"),
            64 => ("long", "ulong"),
            _ => return None,
        };
        match int.signedness() {
            Signedness::Signed => signed,
            _ => unsigned,
        }
    } else if inner.is::<Float16Type>() {
        "half"
    } else if inner.is::<Float32Type>() || inner.is::<FloatFlex32Type>() {
        "float"
    } else if inner.is::<Float64Type>() {
        "double"
    } else {
        return None;
    };
    Some(scalar)
}

opencl_ty!(VectorType, |ty, ctx| {
    native_vector(ctx, ty)
        .unwrap_or_else(|| format!("{}_{}", ty.inner.to_cpp(ctx), ty.vectorization))
});
