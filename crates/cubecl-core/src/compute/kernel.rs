use std::{fmt::Display, marker::PhantomData};

use crate::{codegen::CompilerRepresentation, ir::CubeDim, Compiler, Kernel, KernelId};
use alloc::sync::Arc;
use cubecl_runtime::ExecutionMode;

/// A kernel, compiled in the target language
pub struct CompiledKernel<C: Compiler> {
    pub name: Option<&'static str>,
    /// Source code of the kernel
    pub source: String,
    /// In-memory representation of the kernel
    pub repr: Option<C::Representation>,
    /// Size of a cube for the compiled kernel
    pub cube_dim: CubeDim,
    /// The number of bytes used by the share memory
    pub shared_mem_bytes: usize,
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
                    format!("{:?}", info).as_str(),
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

        if let Some(name) = self.name {
            if name.len() <= 32 {
                f.write_fmt(format_args!("\nname: {name}"))?;
            } else {
                let name = format_str(name, &[('<', '>')], false);
                f.write_fmt(format_args!("\nname: {name}"))?;
            }
        }

        f.write_fmt(format_args!(
            "
cube_dim: ({}, {}, {})
shared_memory: {} bytes",
            self.cube_dim.x, self.cube_dim.y, self.cube_dim.z, self.shared_mem_bytes,
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

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
pub trait CubeTask<C: Compiler>: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;
    /// Compile the kernel into source
    fn compile(&self, mode: ExecutionMode) -> CompiledKernel<C>;
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
    fn compile(&self, mode: ExecutionMode) -> CompiledKernel<C> {
        let gpu_ir = self.kernel_definition.define();
        let cube_dim = gpu_ir.cube_dim;
        let lower_level_ir = C::compile(gpu_ir, mode);
        let shared_mem_bytes = lower_level_ir.shared_memory_size();

        CompiledKernel {
            name: Some(core::any::type_name::<K>()),
            source: lower_level_ir.to_string(),
            repr: Some(lower_level_ir),
            cube_dim,
            shared_mem_bytes,
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
    fn compile(&self, mode: ExecutionMode) -> CompiledKernel<C> {
        self.as_ref().compile(mode)
    }

    fn id(&self) -> KernelId {
        self.as_ref().id()
    }
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }
}

impl<C: Compiler> CubeTask<C> for Box<dyn CubeTask<C>> {
    fn compile(&self, mode: ExecutionMode) -> CompiledKernel<C> {
        self.as_ref().compile(mode)
    }

    fn id(&self) -> KernelId {
        self.as_ref().id()
    }

    fn name(&self) -> &'static str {
        self.as_ref().name()
    }
}
