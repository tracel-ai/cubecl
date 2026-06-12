use cubecl_ir::{AddressSpace, Operation, OperationReflect, Operator, ExpandValue, ValueKind};
use hashbrown::HashMap;
use smallvec::SmallVec;

use crate::Function;

use super::Expression;

impl Expression {
    pub fn to_operation(&self, leaders: &HashMap<u32, ExpandValue>) -> Operation {
        match self {
            Expression::Copy(val, _) => {
                let input = leaders[val];
                Operation::Copy(input)
            }
            Expression::Value(value) | Expression::Volatile(value) => Operation::Copy(*value),
            Expression::Instruction(instruction) => {
                let args = instruction
                    .args
                    .iter()
                    .map(|val| leaders[val])
                    .collect::<SmallVec<[ExpandValue; 4]>>();

                <Operation as OperationReflect>::from_code_and_args(instruction.op, &args).unwrap()
            }
            Expression::Builtin(builtin, _) => Operation::Operator(Operator::ReadBuiltin(*builtin)),
            Expression::Phi(_) => unreachable!("Phi can't be made into operation"),
        }
    }
}

impl Function {
    pub(super) fn value_of_var(&self, var: &ExpandValue) -> Option<ExpandValue> {
        match &var.kind {
            // TODO: This is a hack and should be replaced with instruction-level invalidation
            ValueKind::Value { id }
                if let Some(mem) = self.memories.get(id)
                    && matches!(
                        mem.address_space,
                        AddressSpace::Local | AddressSpace::Shared
                    ) =>
            {
                None
            }
            _ => Some(*var),
        }
    }
}
