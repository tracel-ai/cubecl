use cubecl_core::{
    frontend::{expand_hypot, expand_rhypot},
    ir::{Arithmetic, Instruction, Operation, Scope},
};
use cubecl_opt::{IrTransformer, TransformAction};

#[derive(Debug)]
pub(crate) struct HypotTransform;

impl IrTransformer for HypotTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Hypot(op)) => {
                let mut scope = scope.child();
                expand_hypot(&mut scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(scope.process([]).instructions)
            }
            _ => TransformAction::Ignore,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RhypotTransform;

impl IrTransformer for RhypotTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Rhypot(op)) => {
                let mut scope = scope.child();
                expand_rhypot(&mut scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(scope.process([]).instructions)
            }
            _ => TransformAction::Ignore,
        }
    }
}
