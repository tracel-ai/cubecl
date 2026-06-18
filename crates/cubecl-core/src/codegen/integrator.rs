use alloc::vec::Vec;
use cubecl_ir::{Scope, StorageType, metadata::Info, settings::KernelSettings};
use cubecl_runtime::kernel::{KernelDefinition, ScalarKernelArg};

/// The kernel integrator allows you to create a [kernel definition](KernelDefinition) based on
/// [kernel expansion](KernelExpansion) and [kernel settings](KernelSettings).
#[derive(Debug)]
pub struct KernelIntegrator {
    expansion: KernelExpansion,
    scalar_bindings: Vec<ScalarKernelArg>,
}

/// The information necessary to compile a [kernel definition](KernelDefinition).
#[derive(Debug)]
pub struct KernelExpansion {
    pub scope: Scope,
    pub info: Info,
}

/// Information related to a scalar input.
#[derive(Clone, Debug)]
pub struct ScalarInfo {
    pub ty: StorageType,
    pub count: usize,
}

impl KernelIntegrator {
    /// Starts a new compilation.
    pub fn new(info: KernelExpansion) -> Self {
        Self {
            expansion: info,
            scalar_bindings: Default::default(),
        }
    }

    /// Performs the compilation with the provided [settings](KernelSettings).
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn integrate(mut self, settings: KernelSettings) -> KernelDefinition {
        self.scalar_bindings.sort_by_key(|binding| binding.ty);

        KernelDefinition {
            scalars: self.scalar_bindings,
            body: self.expansion.scope,
            info: self.expansion.info,
            settings,
        }
    }
}
