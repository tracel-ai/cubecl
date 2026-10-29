//! Target-specific lowering.

use crate::{
    prelude::{EntryArgLayout, FuncOp, OpPass, Passes},
    shared::fp4::Fp4Decode,
};

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

    /// How the target decodes `e2m1x2`.
    fn fp4_decode(&self) -> Fp4Decode {
        Fp4Decode::Software
    }
}
