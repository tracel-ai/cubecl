//! The compile step of a launch: what a [`CubeKernel`] becomes once a
//! [`Compiler`] has seen it.

pub use cubecl_runtime::kernel::*;

use alloc::string::{String, ToString};
use core::{
    fmt::Display,
    sync::atomic::{AtomicI8, Ordering},
};

use cubecl_common::format::format_str;
use cubecl_environment::backtrace::BackTrace;

use crate::{
    compiler::{CompilationError, Compiler},
    config::{CubeClRuntimeConfig, RuntimeConfig, compilation::CompilationLogLevel},
    id::KernelId,
    server::CubeDim,
};

/// A kernel, compiled in the target language
pub struct CompiledKernel<C: Compiler> {
    /// The name of the kernel entrypoint.
    /// For example
    ///
    /// ```text
    /// #[cube(launch)]
    /// fn gelu_array<F: Float>() {}
    /// ```
    ///
    /// would have the entrypoint name "`gelu_array`".
    pub entrypoint_name: String,

    /// A fully qualified debug name of the kernel.
    ///
    /// For example
    ///
    /// ```text
    /// #[cube(launch)]
    /// fn gelu_array<F: Float>() {}
    /// ```
    ///
    /// would have a debug name such as
    ///
    /// ```text
    /// gelu::gelu_array::GeluArray<
    ///    cubecl_core::frontend::element::float::F32,
    ///    cubecl_cuda::runtime::CudaRuntime,
    /// >
    /// ```
    pub debug_name: Option<&'static str>,

    /// Source code of the kernel
    pub source: String,
    /// In-memory representation of the kernel
    pub repr: Option<C::Representation>,
    /// Size of a cube for the compiled kernel
    pub cube_dim: CubeDim,
    /// What the kernel does with each buffer binding, by buffer position —
    /// see [`BufferIOAttr`]. `None` when the compiler kept no answer, which the
    /// launch path reads as every buffer both read and written: the
    /// conservative direction, since over-claiming costs a spurious loud
    /// failure and under-claiming costs a silent clean read of garbage.
    pub io: Option<alloc::vec::Vec<BufferIOAttr>>,
    /// Extra debugging information about the compiled kernel.
    pub debug_info: Option<DebugInformation>,
}

/// Extra debugging information about the compiled kernel.
#[derive(new)]
pub struct DebugInformation {
    /// The language tag of the source..
    pub lang_tag: &'static str,
    /// The compilation id.
    pub id: KernelId,
}

impl<C: Compiler> CompiledKernel<C> {
    /// Compile `definition` with `compiler`, keeping `kernel`'s name as the
    /// debug name of the result.
    pub fn compile(
        kernel: &dyn CubeKernel,
        definition: KernelDefinition,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
    ) -> Result<Self, CompilationError> {
        let entrypoint_name = definition.settings.kernel_name.clone();
        let cube_dim = definition.settings.cube_dim.into();

        // A hand-written kernel is already in the target language: there is no
        // IR to hand the compiler, so neither analysis it produces exists.
        // `io: None` reads as every buffer both read and written, which is the
        // conservative direction.
        if let Some(precompiled) = kernel.source() {
            if precompiled.lang != compiler.lang_tag() {
                return Err(CompilationError::Generic {
                    reason: alloc::format!(
                        "kernel `{}` carries {} source, but this compiler expects {}",
                        kernel.name(),
                        precompiled.lang,
                        compiler.lang_tag()
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
            return Ok(CompiledKernel {
                entrypoint_name: precompiled.entrypoint_name,
                debug_name: Some(kernel.name()),
                source: precompiled.source,
                io: None,
                repr: None,
                cube_dim,
                debug_info: None,
            });
        }

        let lower_level_ir = compiler.compile(definition, compilation_options)?;

        Ok(CompiledKernel {
            entrypoint_name,
            debug_name: Some(kernel.name()),
            source: lower_level_ir.to_string(),
            io: C::buffer_io(&lower_level_ir),
            repr: Some(lower_level_ir),
            cube_dim,
            debug_info: None,
        })
    }
}

static COMPILATION_LEVEL: AtomicI8 = AtomicI8::new(-1);

fn compilation_level() -> u8 {
    let compilation_level = COMPILATION_LEVEL.load(Ordering::Relaxed);
    if compilation_level == -1 {
        let val = match CubeClRuntimeConfig::get().compilation.logger.level {
            CompilationLogLevel::Full => 2,
            CompilationLogLevel::Disabled => 0,
            CompilationLogLevel::Basic => 1,
        };

        COMPILATION_LEVEL.store(val, Ordering::Relaxed);
        val as u8
    } else {
        compilation_level as u8
    }
}

impl<C: Compiler> Display for CompiledKernel<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match compilation_level() {
            2 => self.format_full(f),
            _ => self.format_basic(f),
        }
    }
}

impl<C: Compiler> CompiledKernel<C> {
    fn format_basic(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[Compiling kernel]")?;
        if let Some(name) = self.debug_name {
            if name.len() <= 32 {
                f.write_fmt(format_args!(" {name}"))?;
            } else {
                f.write_fmt(format_args!(" {}", name.split('<').next().unwrap_or("")))?;
            }
        }

        Ok(())
    }

    fn format_full(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[START_KERNEL_COMPILATION]")?;

        if let Some(name) = self.debug_name {
            if name.len() <= 32 {
                f.write_fmt(format_args!("\nname: {name}"))?;
            } else {
                let name = format_str(name, &[('<', '>')], false);
                f.write_fmt(format_args!("\nname: {name}"))?;
            }
        }

        if let Some(info) = &self.debug_info {
            f.write_fmt(format_args!("\nid: {:#?}", info.id))?;
        }

        f.write_fmt(format_args!(
            "
source:
```{}
{}
```
[END_KERNEL_COMPILATION]
",
            self.debug_info
                .as_ref()
                .map(|info| info.lang_tag)
                .unwrap_or(""),
            self.source
        ))
    }
}
