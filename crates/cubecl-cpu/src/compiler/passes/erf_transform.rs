use cubecl_core::{
    ir::{Arithmetic, Instruction, Operation, Scope},
    prelude::*,
};
use cubecl_opt::{IrTransformer, TransformAction};

#[derive(Debug)]
pub(crate) struct ErfTransform;

impl IrTransformer for ErfTransform {
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Erf(op)) => {
                let scope = scope.child();
                expand_erf(&scope, op.input, inst.out.unwrap());
                TransformAction::Replace(scope.process([]).instructions)
            }
            _ => TransformAction::Ignore,
        }
    }
}
