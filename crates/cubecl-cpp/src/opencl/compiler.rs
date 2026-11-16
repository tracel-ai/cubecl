use cubecl_core::{ir::ContextExt, prelude::KernelDefinition};
use cubecl_runtime::{
    compiler::{CompilationError, Compiler},
    kernel::BufferIOAttr,
};
use pliron::context::Context;

use crate::{
    shared::{CompilationOptions, ComputeKernel, CppCompiler},
    target::OpenCl,
};

/// What an `OpenCL` device adds to `OpenCL` C 3.0, which the dialect reads while it emits.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenClOptions {
    /// Compile every kernel that uses planes (subgroups) or cooperative matrices for this subgroup
    /// size (`cl_intel_required_subgroup_size`); `None` leaves the size to the driver, and
    /// `PLANE_DIM` is then whatever it chose.
    pub subgroup_size: Option<u32>,
    /// Cooperative matrices on Intel's DPAS (`cl_intel_subgroup_matrix_multiply_accumulate`):
    /// 8-row tiles as wide as the subgroup (`subgroup_size`, 8 or 16).
    pub dpas: bool,
}

impl OpenClOptions {
    /// The options the compilation of `ctx` was given (the defaults outside an
    /// [`OpenClCompiler`] compilation).
    pub fn of(ctx: &Context) -> Self {
        ctx.try_aux_ty::<Self>().copied().unwrap_or_default()
    }
}

/// The options of an `OpenCL` compilation: the shared C++ ones and the device's.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenClCompilationOptions {
    /// The options every C++ target takes.
    pub cpp: CompilationOptions,
    /// What the device adds.
    pub opencl: OpenClOptions,
}

/// Compiles kernels to `OpenCL` C 3.0 for one device's [`OpenClOptions`].
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenClCompiler {
    cpp: CppCompiler<OpenCl>,
}

impl Compiler for OpenClCompiler {
    type Representation = ComputeKernel;
    type CompilationOptions = OpenClCompilationOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        let opencl = options.opencl;
        self.cpp
            .compile_with(kernel, &options.cpp, |ctx: &mut Context| {
                ctx.set_aux_ty(opencl)
            })
    }

    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        <CppCompiler<OpenCl> as Compiler>::buffer_io(repr)
    }

    fn extension(&self) -> &'static str {
        self.cpp.extension()
    }

    fn lang_tag(&self) -> &'static str {
        self.cpp.lang_tag()
    }
}
