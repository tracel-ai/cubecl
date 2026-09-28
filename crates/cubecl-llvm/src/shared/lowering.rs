//! Target-specific lowering.

use crate::prelude::{EntryArgLayout, FuncOp, OpPass, Passes};

/// Target passes and kernel argument layout.
pub trait TargetLowering {
    /// Runs before shared optimizations.
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>);

    /// Runs after shared optimizations and polyfill expansion.
    fn epilogue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        let _ = passes;
    }

    /// Kernel argument layout.
    fn arg_layout(&self) -> Box<dyn EntryArgLayout>;
}
