//! The warning for a kernel that turned float contraction off on a compiler that can't keep the
//! driver from contracting.

use cubecl_ir::{ContextExt as _, prelude::Context, settings::FloatContraction};

/// Warns when the kernel `ctx` holds turned float contraction off, which this compiler can't
/// keep the driver from doing, for the reason `unkept`.
pub fn warn_unkept_no_contraction(ctx: &Context, kernel_name: &str, unkept: &str) {
    if ctx.float_contraction() == FloatContraction::Off {
        log::warn!(
            "[{kernel_name}] {unkept}. Use the SPIR-V compiler, or the cubecl-metal runtime on \
             Apple, for a kernel that must round every operation"
        );
    }
}
