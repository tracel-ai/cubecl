use std::rc::Rc;

use cubecl_common::{CubeDim, ExecutionMode};
use cubecl_ir::{Instruction, Scope};

use crate::Optimizer;

/// Build an optimizer with IR transformers
#[derive(Debug, Default)]
pub struct OptimizerBuilder {
    transformers: Vec<Rc<dyn IrTransformer>>,
}

impl OptimizerBuilder {
    /// Add an IR transformer to the optimizer
    pub fn with_transformer(mut self, transformer: impl IrTransformer + 'static) -> Self {
        self.transformers.push(Rc::new(transformer));
        self
    }

    /// Build and run optimizer on the scope
    pub fn optimize(self, expand: Scope, cube_dim: CubeDim, mode: ExecutionMode) -> Optimizer {
        Optimizer::new(expand, cube_dim, mode, self.transformers)
    }
}

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
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction;
}
