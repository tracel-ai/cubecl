use alloc::{boxed::Box, rc::Rc, vec::Vec};
use cubecl_core::{CubeDim, post_processing::visitor::InstructionVisitor};
use cubecl_ir::{Instruction, Processor, Scope};

use crate::Optimizer;

/// Build an optimizer with IR transformers
#[derive(Default)]
pub struct OptimizerBuilder {
    transformers: Vec<Rc<dyn IrTransformer>>,
    visitors: Vec<Box<dyn InstructionVisitor>>,
    processors: Vec<Box<dyn Processor>>,
}

impl OptimizerBuilder {
    /// Add an IR transformer to the optimizer
    pub fn with_transformer(mut self, transformer: impl IrTransformer + 'static) -> Self {
        self.transformers.push(Rc::new(transformer));
        self
    }

    pub fn with_visitor(mut self, visitor: impl InstructionVisitor + 'static) -> Self {
        self.visitors.push(Box::new(visitor));
        self
    }

    pub fn with_processor(mut self, processor: impl Processor + 'static) -> Self {
        self.processors.push(Box::new(processor));
        self
    }

    /// Build and run optimizer on the scope
    pub fn optimize(self, expand: Scope, cube_dim: CubeDim) -> Optimizer {
        Optimizer::new(
            expand,
            cube_dim,
            self.transformers,
            self.visitors,
            self.processors,
        )
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
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction;
}
