use core::cell::Ref;

use cubecl_ir::{prelude::*, verify_attr_succ};
use thiserror::Error;

#[op_interface]
pub trait OpToWgsl {
    verify_op_succ!();
    fn to_wgsl(&self, ctx: &Context) -> String;
}

#[type_interface]
pub trait TypeToWgsl {
    verify_ty_succ!();
    fn to_wgsl(&self, ctx: &Context) -> String;
}

#[attr_interface]
pub trait AttrToWgsl {
    verify_attr_succ!();
    fn to_wgsl(&self, ctx: &Context) -> String;
}

macro_rules! wgsl_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::compiler::wgsl::to_wgsl::OpToWgsl for $ty {
            fn to_wgsl(&self, ctx: &pliron::context::Context) -> String {
                $crate::compiler::wgsl::to_wgsl::closure_inference_hack::<$ty, String>(
                    self, ctx, $impl,
                )
            }
        }
    };
}
pub(crate) use wgsl_op;

macro_rules! wgsl_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::compiler::wgsl::to_wgsl::OpToWgsl for $ty {
            fn to_wgsl(&self, ctx: &pliron::context::Context) -> String {
                use cubecl_core::ir::prelude::*;
                use $crate::compiler::wgsl::value::WgslValue;
                let op = $crate::compiler::wgsl::to_wgsl::closure_inference_hack::<$ty, String>(
                    self, ctx, $impl,
                );
                let out = self.get_result(ctx).fmt_left(ctx);
                format!("{out} = {op};\n")
            }
        }
    };
}
pub(crate) use wgsl_op_with_out;

pub(crate) fn closure_inference_hack<T, R>(
    val: &T,
    ctx: &Context,
    func: impl FnOnce(&T, &Context) -> R,
) -> R {
    func(val, ctx)
}

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("Encountered unsupported type `{0}`")]
    UnsupportedType(String),
    #[error("Encountered unsupported operation `{0}`")]
    UnsupportedOp(String),
}

pub type Result<T> = core::result::Result<T, CompileError>;

pub trait OpExtWgsl {
    fn to_wgsl(&self, ctx: &Context) -> Result<String>;
}

impl OpExtWgsl for Ptr<Operation> {
    fn to_wgsl(&self, ctx: &Context) -> Result<String> {
        let op_dyn = self.dyn_op(ctx);
        let to_wgsl = op_cast::<dyn OpToWgsl>(op_dyn.as_ref())
            .ok_or_else(|| CompileError::UnsupportedOp(op_dyn.disp(ctx).to_string()))?;
        Ok(to_wgsl.to_wgsl(ctx))
    }
}

pub trait TypeExtWgsl {
    fn to_wgsl(&self, ctx: &Context) -> String;
}
impl TypeExtWgsl for Ref<'_, dyn Type> {
    fn to_wgsl(&self, ctx: &Context) -> String {
        let to_wgsl = type_cast::<dyn TypeToWgsl>(&**self)
            .ok_or_else(|| {
                CompileError::UnsupportedType(format!("{}{}", self.get_type_id(), self.disp(ctx)))
            })
            .unwrap(); // Unwrap for now because handling errors everywhere is annoying
        to_wgsl.to_wgsl(ctx)
    }
}
impl TypeExtWgsl for TypeHandle {
    fn to_wgsl(&self, ctx: &Context) -> String {
        self.deref(ctx).to_wgsl(ctx)
    }
}
impl<T: Type> TypeExtWgsl for TypedHandle<T> {
    fn to_wgsl(&self, ctx: &Context) -> String {
        self.to_handle().to_wgsl(ctx)
    }
}
impl TypeExtWgsl for TypeAttr {
    fn to_wgsl(&self, ctx: &Context) -> String {
        self.get_type(ctx).to_wgsl(ctx)
    }
}
