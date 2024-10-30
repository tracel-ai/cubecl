use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::prelude::*;
use cubecl_core::Feature;

/// Checks if the matmul cmma can be used.
pub fn check_cmma_availability<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
) -> Result<(), ()> {
    if !client.properties().feature_enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: 16,
        k: 16,
        n: 16,
    }) {
        return Err(());
    }

    Ok(())
}
