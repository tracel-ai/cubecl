use cubecl_ir::{ElemType, Scope, metadata::Info, settings::KernelSettings};
use cubecl_runtime::kernel::KernelDefinition;

/// The kernel integrator allows you to create a [kernel definition](KernelDefinition) based on
/// [kernel expansion](KernelExpansion) and [kernel settings](KernelSettings).
#[derive(Debug)]
pub struct KernelIntegrator {
    expansion: KernelExpansion,
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
    pub ty: ElemType,
    pub count: usize,
}

impl KernelIntegrator {
    /// Starts a new compilation.
    pub fn new(info: KernelExpansion) -> Self {
        Self { expansion: info }
    }

    /// Performs the compilation with the provided [settings](KernelSettings).
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn integrate(self, settings: KernelSettings) -> KernelDefinition {
        KernelDefinition {
            body: self.expansion.scope,
            info: self.expansion.info,
            settings,
        }
    }
}
