use cubecl_core::ir::{
    ConstantScalarValue, Instruction, IntKind, Operation, Operator, Scope, Variable, VariableKind,
};
use cubecl_opt::{IrTransformer, TransformAction};

#[derive(Debug)]
pub struct BuiltinReplace;

impl BuiltinReplace {
    pub fn replace_variable(&self, variable: &mut Variable) -> bool {
        match variable.kind {
            VariableKind::Builtin(_builtin) => {
                *variable = Variable::constant(ConstantScalarValue::Int(0, IntKind::I64));
                true
            }
            _ => false,
        }
    }
}

impl IrTransformer for BuiltinReplace {
    fn maybe_transform(&self, _scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Operator(Operator::Index(index)) => {
                let mut index = index.clone();
                match self.replace_variable(&mut index.index) {
                    true => {
                        let index: Operator = Operator::Index(index);
                        TransformAction::Replace(vec![Instruction::new(index, inst.out.unwrap())])
                    }
                    false => TransformAction::Ignore,
                }
            }
            Operation::Operator(Operator::IndexAssign(index_assign)) => {
                let mut index_assign = index_assign.clone();
                match self.replace_variable(&mut index_assign.index) {
                    true => {
                        let index_assign: Operator = Operator::IndexAssign(index_assign);
                        TransformAction::Replace(vec![Instruction::new(
                            index_assign,
                            inst.out.unwrap(),
                        )])
                    }
                    false => TransformAction::Ignore,
                }
            }
            _ => TransformAction::Ignore,
        }
    }
}
