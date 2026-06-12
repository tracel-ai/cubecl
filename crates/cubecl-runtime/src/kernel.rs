use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use core::{
    fmt::Display,
    hash::Hash,
    marker::PhantomData,
    sync::atomic::{AtomicI8, Ordering},
};

use cubecl_common::{
    format::format_str,
    hash::{StableHash, StableHasher},
};
use cubecl_ir::{Scope, StorageType, pliron::value::Value};
use serde::{Deserialize, Serialize};

use crate::{
    compiler::{CompilationError, Compiler, CubeTask},
    config::{CubeClRuntimeConfig, RuntimeConfig, compilation::CompilationLogLevel},
    id::KernelId,
    server::{CubeDim, ExecutionMode},
};

/// Implement this trait to create a [kernel definition](KernelDefinition).
pub trait KernelMetadata: Send + Sync + 'static {
    /// Name of the kernel for debugging.
    fn name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }

    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;

    /// Type of addresses in this kernel
    fn address_type(&self) -> StorageType;
}

#[allow(missing_docs)]
pub struct KernelDefinition {
    pub buffers: Vec<KernelArg>,
    pub tensor_maps: Vec<KernelArg>,
    pub scalars: Vec<ScalarKernelArg>,
    pub cube_dim: CubeDim,
    pub body: Scope,
    pub options: KernelOptions,
}

impl KernelDefinition {
    /// Returns the total number of global buffers (including tensor maps)
    pub fn num_global_buffers(&self) -> usize {
        self.buffers.len() + self.tensor_maps.len()
    }

    /// Hash the content of the kernel in a stable way that can be used between runs.
    ///
    /// Two kernels with the same hash expand to the same IR, so a compiled artifact keyed on it
    /// stays valid exactly as long as the code producing it is unchanged. This is what makes the
    /// persistent compilation cache pick up edits to a kernel body, or to any `#[cube]` function it
    /// reaches, without relying on a version bump.
    ///
    /// Debug information is deliberately left out: it holds absolute source paths, which would make
    /// the hash differ between machines and checkouts, and it never changes what the compiler emits
    /// beyond [`KernelOptions::debug_symbols`], which is hashed.
    pub fn stable_hash(&self) -> StableHash {
        let mut hasher = StableHasher::new();

        self.buffers.hash(&mut hasher);
        self.tensor_maps.hash(&mut hasher);
        self.scalars.hash(&mut hasher);
        self.cube_dim.hash(&mut hasher);
        self.options.hash(&mut hasher);
        self.body.hash(&mut hasher);

        // `Scope` skips the global state when hashing, so outlined functions have to be hashed
        // here. They aren't reachable from the body instructions, which only reference them by id,
        // meaning a change confined to one of them would otherwise go unnoticed. The map is ordered
        // by id, so the traversal is deterministic.
        let state = self.body.state();
        for (id, function) in state.functions.iter() {
            id.hash(&mut hasher);
            function.hash(&mut hasher);
        }

        hasher.finalize()
    }
}

#[derive(Default, Clone, Debug, Hash, PartialEq, Eq)]
/// Options for a specific kernel compilation
pub struct KernelOptions {
    /// The name of the kernel
    pub kernel_name: String,
    /// Whether to include debug symbols
    pub debug_symbols: bool,
    /// CUDA Cluster dim, if any
    pub cluster_dim: Option<CubeDim>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
/// Global argument of a kernel.
pub struct KernelArg {
    /// The index of the arg.
    pub id: usize,
    /// The value the argument is bound to.
    pub value: Value,
    /// Whether the argument has metadata.
    pub has_extended_meta: bool,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarKernelArg {
    pub ty: StorageType,
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub enum Visibility {
    Uniform,
    Read,
    ReadWrite,
}

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
    /// would have the entrypoint name "`gelu_array`".
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

/// Kernel that can be defined
pub trait CubeKernel: KernelMetadata {
    /// Define the kernel for compilation
    fn define(&self) -> KernelDefinition;
}

/// Wraps a [`CubeKernel`] to allow it be compiled.
pub struct KernelTask<C: Compiler, K: CubeKernel> {
    kernel_definition: K,
    _compiler: PhantomData<C>,
}

/// Generic [`CubeTask`] for compiling kernels
pub struct CubeTaskKernel<C: Compiler> {
    /// The inner compilation task being wrapped
    pub task: Box<dyn CubeTask<C>>,
}

impl<C: Compiler, K: CubeKernel> KernelTask<C, K> {
    /// Create a new kernel task
    pub fn new(kernel_definition: K) -> Self {
        Self {
            kernel_definition,
            _compiler: PhantomData,
        }
    }
}

impl<C: Compiler, K: CubeKernel> CubeTask<C> for KernelTask<C, K> {
    fn define(&self) -> KernelDefinition {
        self.kernel_definition.define()
    }

    fn compile(
        &self,
        gpu_ir: KernelDefinition,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let entrypoint_name = gpu_ir.options.kernel_name.clone();
        let cube_dim = gpu_ir.cube_dim;
        let lower_level_ir = compiler.compile(gpu_ir, compilation_options, mode, addr_type)?;

        Ok(CompiledKernel {
            entrypoint_name,
            debug_name: Some(core::any::type_name::<K>()),
            source: lower_level_ir.to_string(),
            repr: Some(lower_level_ir),
            cube_dim,
            debug_info: None,
        })
    }
}

impl<C: Compiler, K: CubeKernel> KernelMetadata for KernelTask<C, K> {
    // Forward ID to underlying kernel definition.
    fn id(&self) -> KernelId {
        self.kernel_definition.id()
    }

    // Forward name to underlying kernel definition.
    fn name(&self) -> &'static str {
        self.kernel_definition.name()
    }

    fn address_type(&self) -> StorageType {
        self.kernel_definition.address_type()
    }
}

impl<C: Compiler> KernelMetadata for Box<dyn CubeTask<C>> {
    // Deref and use existing ID.
    fn id(&self) -> KernelId {
        self.as_ref().id()
    }

    // Deref and use existing name.
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    fn address_type(&self) -> StorageType {
        self.as_ref().address_type()
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_ir::{ElemType, Instruction, Operation, Type};

    fn definition(body: Scope) -> KernelDefinition {
        KernelDefinition {
            buffers: Vec::new(),
            tensor_maps: Vec::new(),
            scalars: Vec::new(),
            cube_dim: CubeDim::new_single(),
            body,
            options: KernelOptions::default(),
        }
    }

    /// A scope holding a single `Copy` on a freshly declared local.
    fn scope_with_copy() -> Scope {
        let scope = Scope::root(false);
        let local = scope.create_local_mut(Type::scalar(ElemType::Bool));
        scope.register(Instruction::new(Operation::Copy(local), local));
        scope
    }

    #[test]
    fn hash_is_stable_across_calls() {
        let definition = definition(scope_with_copy());

        assert_eq!(definition.stable_hash(), definition.stable_hash());
    }

    #[test]
    fn equivalent_definitions_hash_equal() {
        let lhs = definition(scope_with_copy());
        let rhs = definition(scope_with_copy());

        assert_eq!(lhs.stable_hash(), rhs.stable_hash());
    }

    #[test]
    fn body_change_changes_hash() {
        let lhs = definition(scope_with_copy());

        let scope = Scope::root(false);
        let local = scope.create_local_mut(Type::scalar(ElemType::Bool));
        // Same shape as `scope_with_copy`, different operation.
        scope.register(Instruction::new(
            Operation::ConstructAggregate(alloc::vec![local]),
            local,
        ));
        let rhs = definition(scope);

        assert_ne!(lhs.stable_hash(), rhs.stable_hash());
    }

    #[test]
    fn cube_dim_change_changes_hash() {
        let lhs = definition(scope_with_copy());
        let mut rhs = definition(scope_with_copy());
        rhs.cube_dim = CubeDim::new_2d(2, 2);

        assert_ne!(lhs.stable_hash(), rhs.stable_hash());
    }

    /// The body only references an outlined function by id, and `Scope`'s own `Hash` skips the
    /// global state holding it. Without hashing that map, editing a `#[cube]` helper that got
    /// outlined would leave the key untouched and the stale artifact would be served.
    #[test]
    fn outlined_function_change_changes_hash() {
        // The kernel body is empty either way; only the outlined function differs.
        fn with_function(extra_instruction: bool) -> KernelDefinition {
            let outlined = Scope::root(false);
            let local = outlined.create_local_mut(Type::scalar(ElemType::Bool));
            outlined.register(Instruction::new(Operation::Copy(local), local));
            if extra_instruction {
                outlined.register(Instruction::new(
                    Operation::ConstructAggregate(alloc::vec![local]),
                    local,
                ));
            }

            let definition = definition(Scope::root(false));
            definition.body.create_function(Vec::new(), outlined);
            definition
        }

        let lhs = with_function(false);
        let rhs = with_function(true);

        // The bodies are indistinguishable on their own; only hashing the outlined function
        // separates the two definitions.
        assert_eq!(
            StableHasher::hash_one(&lhs.body),
            StableHasher::hash_one(&rhs.body)
        );
        assert_ne!(lhs.stable_hash(), rhs.stable_hash());
    }
}
