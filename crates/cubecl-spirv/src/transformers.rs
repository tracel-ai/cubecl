use cubecl_core::{
    ir::{Arithmetic, Instruction, Operation, Scope},
    prelude::expand_erf,
};
use cubecl_opt::{IrTransformer, TransformAction};

#[derive(Debug)]
pub(crate) struct ErfTransform;

impl IrTransformer for ErfTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Erf(op)) => {
                let mut scope = scope.child();
                expand_erf(&mut scope, op.input, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

fn into_instructions(mut scope: Scope) -> Vec<Instruction> {
    scope.process().operations
}
