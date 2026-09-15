//! Target-specific lowering.

use pliron::builtin::ops::FuncOp;
use pliron::pass::{OpPass, Passes};

use crate::shared::metadata::EntryArgLayout;

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
