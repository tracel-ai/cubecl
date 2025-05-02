use std::{fmt::Display, marker::PhantomData};

use crate::{Compiler, Kernel, KernelId, KernelOptions};
use alloc::sync::Arc;
use cubecl_common::{CubeDim, ExecutionMode};
use cubecl_ir::{Elem, Id, Item, Scope};
use serde::{Deserialize, Serialize};

/// A kernel, compiled in the target language
pub struct CompiledKernel<C: Compiler> {
    /// The name of the kernel entrypoint.
    /// For example
    ///
    /// ```text
    /// #[cube(launch)]
    /// fn gelu_array<F: Float, R: Runtime>() {}
    /// ```
    ///
    /// would have the entrypoint name "gelu_array".
    pub entrypoint_name: String,

    /// A fully qualified debug name of the kernel.
    ///
    /// For example
    ///
    /// ```text
    /// #[cube(launch)]
    /// fn gelu_array<F: Float, R: Runtime>() {}
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

impl Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.info {
            Some(info) => f.write_str(
                format_str(
                    format!("{info:?}").as_str(),
                    &[('(', ')'), ('[', ']'), ('{', '}')],
                    true,
                )
                .as_str(),
            ),
            None => f.write_str("No info"),
        }
    }
}

impl<C: Compiler> Display for CompiledKernel<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("\n[START_KERNEL_COMPILATION]")?;

        if let Some(name) = self.debug_name {
            if name.len() <= 32 {
                f.write_fmt(format_args!("\nname: {name}"))?;
            } else {
                let name = format_str(name, &[('<', '>')], false);
                f.write_fmt(format_args!("\nname: {name}"))?;
            }
        }

        f.write_fmt(format_args!(
            "
cube_dim: ({}, {}, {})",
            self.cube_dim.x, self.cube_dim.y, self.cube_dim.z,
        ))?;

        if let Some(info) = &self.debug_info {
            f.write_fmt(format_args!(
                "\ninfo: {}",
                format_str(
                    format!("{:?}", info.id).as_str(),
                    &[('(', ')'), ('[', ']'), ('{', '}')],
                    true
                )
            ))?;
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

fn format_str(kernel_id: &str, markers: &[(char, char)], include_space: bool) -> String {
    let kernel_id = kernel_id.to_string();
    let mut result = String::new();
    let mut depth = 0;
    let indentation = 4;

    let mut prev = ' ';

    for c in kernel_id.chars() {
        if c == ' ' {
            continue;
        }

        let mut found_marker = false;

        for (start, end) in markers {
            let (start, end) = (*start, *end);

            if c == start {
                depth += 1;
                if prev != ' ' && include_space {
                    result.push(' ');
                }
                result.push(start);
                result.push('\n');
                result.push_str(&" ".repeat(indentation * depth));
                found_marker = true;
            } else if c == end {
                depth -= 1;
                if prev != start {
                    if prev == ' ' {
                        result.pop();
                    }
                    result.push_str(",\n");
                    result.push_str(&" ".repeat(indentation * depth));
                    result.push(end);
                } else {
                    for _ in 0..(&" ".repeat(indentation * depth).len()) + 1 + indentation {
                        result.pop();
                    }
                    result.push(end);
                }
                found_marker = true;
            }
        }

        if found_marker {
            prev = c;
            continue;
        }

        if c == ',' && depth > 0 {
            if prev == ' ' {
                result.pop();
            }

            result.push_str(",\n");
            result.push_str(&" ".repeat(indentation * depth));
            continue;
        }

        if c == ':' && include_space {
            result.push(c);
            result.push(' ');
            prev = ' ';
        } else {
            result.push(c);
            prev = c;
        }
    }

    result
}

#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct KernelDefinition {
    pub buffers: Vec<Binding>,
    pub tensor_maps: Vec<Id>,
    pub scalars: Vec<ScalarBinding>,
    pub cube_dim: CubeDim,
    pub body: Scope,
    pub options: KernelOptions,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Binding {
    pub id: Id,
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
    pub has_extended_meta: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarBinding {
    pub elem: Elem,
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Location {
    Storage,
    Cube,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Visibility {
    Read,
    ReadWrite,
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
pub trait CubeTask<C: Compiler>: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;
    /// Compile the kernel into source
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
    ) -> CompiledKernel<C>;
    fn name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }
}

/// Wraps a [kernel](Kernel) to create a [cube task](CubeTask).
#[derive(new)]
pub struct KernelTask<C: Compiler, K: Kernel> {
    kernel_definition: K,
    _compiler: PhantomData<C>,
}

impl<C: Compiler, K: Kernel> CubeTask<C> for KernelTask<C, K> {
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
    ) -> CompiledKernel<C> {
        let gpu_ir = self.kernel_definition.define();
        let entrypoint_name = gpu_ir.options.kernel_name.clone();
        let cube_dim = gpu_ir.cube_dim;
        let lower_level_ir = compiler.compile(gpu_ir, compilation_options, mode);

        CompiledKernel {
            entrypoint_name,
            debug_name: Some(core::any::type_name::<K>()),
            source: lower_level_ir.to_string(),
            repr: Some(lower_level_ir),
            cube_dim,
            debug_info: None,
        }
    }

    fn id(&self) -> KernelId {
        self.kernel_definition.id().clone()
    }

    fn name(&self) -> &'static str {
        core::any::type_name::<K>()
    }
}

impl<C: Compiler> CubeTask<C> for Arc<dyn CubeTask<C>> {
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
    ) -> CompiledKernel<C> {
        self.as_ref().compile(compiler, compilation_options, mode)
    }

    fn id(&self) -> KernelId {
        self.as_ref().id()
    }
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }
}

impl<C: Compiler> CubeTask<C> for Box<dyn CubeTask<C>> {
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
    ) -> CompiledKernel<C> {
        self.as_ref().compile(compiler, compilation_options, mode)
    }

    fn id(&self) -> KernelId {
        self.as_ref().id()
    }

    fn name(&self) -> &'static str {
        self.as_ref().name()
    }
}
