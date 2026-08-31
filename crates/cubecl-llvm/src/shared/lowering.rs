//! What one target contributes to the pass pipeline the two share.

use pliron::builtin::ops::FuncOp;
use pliron::pass::{OpPass, Passes};

use crate::shared::metadata::EntryArgLayout;

/// The passes around the shared ones, and the entry layout they lower to.
///
/// The optimizations in the middle are the same list for both targets, and drifting apart is
/// the failure mode this exists to remove: one target growing a pass the other silently does
/// without is a difference nothing in the code states.
pub trait TargetLowering {
    /// Runs before the shared optimizations. Where the target establishes its memory model
    /// and its launch grid, both of which the optimizations then see.
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>);

    /// Runs after them, once the polyfills that read builtins of their own have been
    /// expanded. The CPU adds nothing: its grid became a loop nest in the prologue.
    fn epilogue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        let _ = passes;
    }

    /// How the entry point presents its arguments to the host.
    fn arg_layout(&self) -> Box<dyn EntryArgLayout>;
}
