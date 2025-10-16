use cubecl_core::prelude::CubePrimitive;

/// Run an arbitrary function with the quantization types from the scheme.
/// Useful when concrete types aren't available.
pub trait RunWithQuantType {
    type Output;

    fn execute<Q: CubePrimitive, S: CubePrimitive>(self) -> Self::Output;
}
