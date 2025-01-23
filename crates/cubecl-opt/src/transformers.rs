use std::rc::Rc;

use cubecl_common::{CubeDim, ExecutionMode};
use cubecl_ir::{Instruction, Scope};

use crate::Optimizer;

#[derive(Debug, Default)]
pub struct OptimizerBuilder {
    transformers: Vec<Rc<dyn IrTransformer>>,
}

impl OptimizerBuilder {
    pub fn with_transformer(mut self, transformer: impl IrTransformer + 'static) -> Self {
        self.transformers.push(Rc::new(transformer));
        self
    }

    pub fn optimize(self, expand: Scope, cube_dim: CubeDim, mode: ExecutionMode) -> Optimizer {
        Optimizer::new(expand, cube_dim, mode, self.transformers)
    }
}

pub enum TransformAction {
    Ignore,
    Replace(Vec<Instruction>),
    Remove,
}

pub trait IrTransformer: core::fmt::Debug {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction;
}
