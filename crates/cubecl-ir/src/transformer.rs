use crate::{Instruction, Scope};

/// The action that should be performed on an instruction, returned by [`IrTransformer::maybe_transform`]
pub enum TransformAction {
    /// The transformer doesn't apply to this instruction
    Ignore,
    /// Replace this instruction with one or more other instructions
    Replace(Vec<Instruction>),
    /// Remove this instruction with no substitute (i.e. debug info)
    Remove,
}

/// A transformer that can modify instructions before they get added to the control flow graph.
pub trait IrTransformer: core::fmt::Debug {
    /// Inspect an instruction and potentially transform it.
    fn maybe_transform(&mut self, scope: &mut Scope, inst: &Instruction) -> TransformAction;
}
