use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use core::{
    fmt::Display,
    marker::PhantomData,
    sync::atomic::{AtomicI8, Ordering},
};

use cubecl_common::{format::format_str, hash::StableHasher};
use cubecl_ir::{Id, Scope, StorageType, Type};
use serde::{Deserialize, Serialize};

use crate::{
    compiler::{CompilationError, Compiler, CubeTask},
    config::{GlobalConfig, compilation::CompilationLogLevel},
    id::KernelId,
    server::{CubeDim, ExecutionMode},
};

#[derive(Debug, Clone)]
/// A version of [`KernelId`] that is faster to hash.
///
/// The expensive hash is precomputed once via [`StableHasher`] and cached in the struct.
/// When used in a [`HashMap`]/[`HashSet`], this avoids recomputing the hash on every lookup.
///
/// # Invariants
///
/// - [`StableHasher`] must be deterministic: the same [`KernelId`] must always produce the
///   same hash. If the hasher were seeded per-process or otherwise non-deterministic, two
///   equal `KernelId`s could yield different hashes, violating the `Hash`/`Eq` contract.
///
/// # Performance Note
///
/// For maximum benefit, use this with an identity/passthrough `BuildHasher` that returns
/// the precomputed hash directly, rather than the default `SipHash` which would hash the
/// `u128` again, partially defeating the purpose of precomputation.
pub struct KernelIdFast {
    /// The hash is expensive to compute so we have it here and reuse it in potentially multiple
    /// maps.
    pub(crate) hash: u128,
    /// The original kernel id.
    pub id: KernelId,
}

impl KernelIdFast {
    /// Creates a new [`KernelIdFast`] by precomputing the hash of the given [`KernelId`].
    pub fn new(kernel_id: KernelId) -> Self {
        let hash = StableHasher::hash_one(&kernel_id);

        Self {
            hash,
            id: kernel_id,
        }
    }
}

impl core::hash::Hash for KernelIdFast {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // Forward the precomputed hash to the hasher. Note that this means the map's
        // hasher will hash this `u128` again — consider pairing with an identity hasher
        // to avoid the double-hash overhead.
        self.hash.hash(state);
    }
}

impl Eq for KernelIdFast {}
impl PartialEq for KernelIdFast {
    fn eq(&self, other: &Self) -> bool {
        // Equality is based on the underlying `KernelId`, not the cached hash.
        // This is correct: two different `KernelId`s may collide on `hash` (pigeonhole
        // principle), so we must compare the actual identifiers for correctness.
        self.id == other.id
    }
}

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

#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct KernelDefinition {
    pub buffers: Vec<KernelArg>,
    pub tensor_maps: Vec<KernelArg>,
    pub scalars: Vec<ScalarKernelArg>,
    pub cube_dim: CubeDim,
    pub body: Scope,
    pub options: KernelOptions,
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
/// Global argument of a kernel.
pub struct KernelArg {
    /// The kernel id.
    pub id: Id,
    /// Where the argument is available.
    pub location: Location,
    /// Whether the global argument can only be accessed for reading, or if it can also be accessed
    /// for write.
    pub visibility: Visibility,
    /// The type of the argument.
    pub ty: Type,
    /// The size of the argument.
    pub size: Option<usize>,
    /// Whether the argument has metadata.
    pub has_extended_meta: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarKernelArg {
    pub ty: StorageType,
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
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let gpu_ir = self.kernel_definition.define();
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
        let val = match GlobalConfig::get().compilation.logger.level {
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
