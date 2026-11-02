//! Target-specific lowering.

use pliron::builtin::ops::ModuleOp;

use crate::{
    prelude::{EntryArgLayout, FuncOp, OpPass, Passes},
    shared::native_decode::NativeDecode,
};

/// Target passes and kernel argument layout.
pub trait TargetLowering {
    /// Runs before shared optimizations.
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>);

    /// Adds the preferred PRE method, which is target specific
    fn pre(&self, passes: &mut OpPass<ModuleOp, Passes>);

    /// Runs after shared optimizations and polyfill expansion.
    fn epilogue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        let _ = passes;
    }

    /// Kernel argument layout.
    fn arg_layout(&self) -> Box<dyn EntryArgLayout>;

    /// The minifloats the target decodes into floats itself ([`NativeDecode`]).
    fn native_decodes(&self) -> &'static [NativeDecode] {
        &[]
    }
}
