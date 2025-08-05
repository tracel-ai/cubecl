use cubecl_common::{CubeDim, ExecutionMode, stub::Mutex};
use cubecl_ir::{Scope, transformer::IrTransformer};

use crate::Optimizer;

/// Build an optimizer with IR transformers
#[derive(Debug, Default)]
pub struct OptimizerBuilder {
    transformers: Vec<Mutex<Box<dyn IrTransformer>>>,
}

impl OptimizerBuilder {
    /// Add an IR transformer to the optimizer
    pub fn with_transformer(mut self, transformer: impl IrTransformer + 'static) -> Self {
        self.transformers.push(Mutex::new(Box::new(transformer)));
        self
    }

    /// Build and run optimizer on the scope
    pub fn optimize(self, expand: Scope, cube_dim: CubeDim, mode: ExecutionMode) -> Optimizer {
        Optimizer::new(expand, cube_dim, mode, self.transformers)
    }
}
