use cubecl_core::{
    frontend::{expand_hypot, expand_rhypot},
    ir::{Arithmetic, Instruction, Operation, Scope},
};
use cubecl_opt::{IrTransformer, TransformAction};

#[derive(Debug)]
pub(crate) struct HypotTransform;

impl IrTransformer for HypotTransform {
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Hypot(op)) => {
                let scope = scope.child();
                expand_hypot(&scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(scope.process([]).instructions)
            }
            _ => TransformAction::Ignore,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RhypotTransform;

impl IrTransformer for RhypotTransform {
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Rhypot(op)) => {
                let scope = scope.child();
                expand_rhypot(&scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(scope.process([]).instructions)
            }
            _ => TransformAction::Ignore,
        }
    }
}
