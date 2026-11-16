use core::fmt::Debug;

use cubecl_core::ir::ContextExt;
use pliron::{context::Context, r#type::Typed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cuda,
    Hip,
    Metal,
    OpenCl,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Shared;
#[derive(Debug, Clone, Copy, Default)]
pub struct Cuda;
#[derive(Debug, Clone, Copy, Default)]
pub struct Hip;
#[derive(Debug, Clone, Copy, Default)]
pub struct Metal;

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCl;

impl Target {
    pub fn ty_prefix(&self, ctx: &Context, ty: impl Typed) -> &'static str {
        if ty.is_half(ctx) {
            self.half_prefix()
        } else if ty.is_half2(ctx) {
            self.half2_prefix()
        } else {
            ""
        }
    }

    pub fn half_prefix(&self) -> &'static str {
        match self {
            Target::Cuda | Target::Hip => "h",
            Target::Metal | Target::OpenCl => "",
        }
    }

    pub fn half2_prefix(&self) -> &'static str {
        match self {
            Target::Cuda | Target::Hip => "h2",
            Target::Metal | Target::OpenCl => "",
        }
    }
}

pub trait CppTarget: Default + Clone + Copy + Debug + Send + Sync + 'static {
    fn target() -> Target;
}

impl CppTarget for Cuda {
    fn target() -> Target {
        Target::Cuda
    }
}
impl CppTarget for Hip {
    fn target() -> Target {
        Target::Hip
    }
}
impl CppTarget for Metal {
    fn target() -> Target {
        Target::Metal
    }
}

impl CppTarget for OpenCl {
    fn target() -> Target {
        Target::OpenCl
    }
}

impl CtxTarget for Context {}
pub trait CtxTarget: ContextExt {
    fn target(&self) -> Target {
        *self.aux_ty::<Target>()
    }
    fn set_target(&mut self, value: Target) {
        self.set_aux_ty(value);
    }
}

macro_rules! dispatch_target {
    ($ctx: expr, $expr: expr) => {{
        use $crate::target::CtxTarget;
        match $ctx.target() {
            $crate::target::Target::Cuda => {
                type Target = $crate::target::Cuda;
                $expr
            }
            $crate::target::Target::Hip => {
                type Target = $crate::target::Hip;
                $expr
            }
            $crate::target::Target::Metal => {
                type Target = $crate::target::Metal;
                $expr
            }
            $crate::target::Target::OpenCl => {
                type Target = $crate::target::OpenCl;
                $expr
            }
        }
    }};
}
pub(crate) use dispatch_target;

use crate::shared::ty::TypedExtCPP;
