use crate::id::KernelId;

/// Implement this trait to create a [kernel definition](KernelDefinition).
pub trait KernelMetadata: Send + Sync + 'static {
    /// Name of the kernel for debugging.
    fn name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }

    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;
}
