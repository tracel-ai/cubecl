use cubecl_core::ir::{Instruction, Operation, Operator, Scope};
use cubecl_opt::{IrTransformer, TransformAction};

#[derive(Debug)]
pub(crate) struct CheckedTransform;

impl IrTransformer for CheckedTransform {
    fn maybe_transform(&self, _scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Operator(Operator::UncheckedIndex(index)) => {
                TransformAction::Replace(vec![Instruction::new(
                    Operator::Index(index.clone()),
                    inst.out.unwrap(),
                )])
            }
            Operation::Operator(Operator::UncheckedIndexAssign(index)) => {
                TransformAction::Replace(vec![Instruction::new(
                    Operator::IndexAssign(index.clone()),
                    inst.out.unwrap(),
                )])
            }
            _ => TransformAction::Ignore,
        }
    }
}
