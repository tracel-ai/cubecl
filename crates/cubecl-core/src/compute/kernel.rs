use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

use crate::{codegen::CompilerRepresentation, ir::CubeDim, Compiler, Kernel};
use alloc::sync::Arc;
use cubecl_runtime::server::{Binding, ComputeServer};

/// A kernel, compiled in the target language
pub struct CompiledKernel {
    pub name: Option<&'static str>,
    /// Source code of the kernel
    pub source: String,
    /// Size of a cube for the compiled kernel
    pub cube_dim: CubeDim,
    /// The number of bytes used by the share memory
    pub shared_mem_bytes: usize,
}

impl Display for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("\n======== Compiled Kernel ========")?;

        if let Some(name) = self.name {
            if name.len() <= 32 {
                f.write_fmt(format_args!("\nname: {name}"))?;
            } else {
                let name = format_type_name(name);
                f.write_fmt(format_args!("\nname: {name}"))?;
            }
        }

        f.write_fmt(format_args!(
            "
cube_dim: ({}, {}, {})
shared_memory: {} bytes
source:
```
{}
```
=================================
",
            self.cube_dim.x, self.cube_dim.y, self.cube_dim.z, self.shared_mem_bytes, self.source
        ))
    }
}

fn format_type_name(type_name: &str) -> String {
    let mut result = String::new();
    let mut depth = 0;
    let indendation = 4;

    for c in type_name.chars() {
        if c == ' ' {
            continue;
        }

        if c == '<' {
            depth += 1;
            result.push_str("<\n");
            result.push_str(&" ".repeat(indendation * depth));
            continue;
        } else if c == '>' {
            depth -= 1;
            result.push_str(",\n>");
            continue;
        }

        if c == ',' && depth > 0 {
            result.push_str(",\n");
            result.push_str(&" ".repeat(indendation * depth));
        } else {
            result.push(c);
        }
    }

    result
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
pub trait CubeTask: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Compile the kernel into source
    fn compile(&self) -> CompiledKernel;
}

/// Wraps a [kernel](Kernel) to create a [cube task](CubeTask).
#[derive(new)]
pub struct KernelTask<C: Compiler, K: Kernel> {
    kernel_definition: K,
    _compiler: PhantomData<C>,
}

impl<C: Compiler, K: Kernel> CubeTask for KernelTask<C, K> {
    fn compile(&self) -> CompiledKernel {
        let gpu_ir = self.kernel_definition.define();
        let cube_dim = gpu_ir.cube_dim;
        let lower_level_ir = C::compile(gpu_ir);
        let shared_mem_bytes = lower_level_ir.shared_memory_size();
        let source = lower_level_ir.to_string();

        CompiledKernel {
            name: Some(core::any::type_name::<K>()),
            source,
            cube_dim,
            shared_mem_bytes,
        }
    }

    fn id(&self) -> String {
        self.kernel_definition.id().clone()
    }
}

impl CubeTask for Arc<dyn CubeTask> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }
}

impl CubeTask for Box<dyn CubeTask> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }
}

/// Provides launch information specifying the number of work groups to be used by a compute shader.
pub enum CubeCount<S: ComputeServer> {
    /// Dispatch x,y,z work groups.
    Static(u32, u32, u32),
    /// Dispatch work groups based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(Binding<S>),
}

impl<S: ComputeServer> Debug for CubeCount<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CubeCount::Static(x, y, z) => f.write_fmt(format_args!("({x}, {y}, {z})")),
            CubeCount::Dynamic(_) => f.write_str("binding"),
        }
    }
}

impl<S: ComputeServer> Clone for CubeCount<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Static(x, y, z) => Self::Static(*x, *y, *z),
            Self::Dynamic(handle) => Self::Dynamic(handle.clone()),
        }
    }
}
