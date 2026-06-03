use cubecl_ir::{Operation, OperationReflect, Operator, Variable, VariableKind};
use hashbrown::HashMap;
use smallvec::SmallVec;

use super::{Expression, Local, Value};

impl Expression {
    pub fn to_operation(&self, leaders: &HashMap<u32, Value>) -> Operation {
        match self {
            Expression::Copy(val, _) => {
                let input = leaders[val].as_var();
                Operation::Copy(input)
            }
            Expression::Value(value) | Expression::Volatile(value) => {
                Operation::Copy(value.as_var())
            }
            Expression::Instruction(instruction) => {
                let args = instruction
                    .args
                    .iter()
                    .map(|val| leaders[val].as_var())
                    .collect::<SmallVec<[Variable; 4]>>();

                <Operation as OperationReflect>::from_code_and_args(instruction.op, &args).unwrap()
            }
            Expression::Builtin(builtin, _) => Operation::Operator(Operator::ReadBuiltin(*builtin)),
            Expression::Phi(_) => unreachable!("Phi can't be made into operation"),
        }
    }
}

impl Value {
    pub(crate) fn as_var(&self) -> Variable {
        match self {
            Value::Constant(val, ty) => Variable::constant(*val, *ty),
            Value::Local(Local { id, item }) => {
                Variable::new(VariableKind::LocalConst { id: *id }, *item)
            }
            Value::Global(id, item) => Variable::new(VariableKind::GlobalBuffer(*id), *item),
        }
    }
}

pub fn value_of_var(var: &Variable) -> Option<Value> {
    let item = var.ty;
    let val = match var.kind {
        VariableKind::GlobalBuffer(id) => Value::Global(id, item),
        VariableKind::LocalConst { id } => Value::Local(Local { id, item }),
        VariableKind::Constant(val) => Value::Constant(val, item),
        VariableKind::LocalMut { .. } | VariableKind::Shared { .. } => None?,
        VariableKind::BarrierToken { .. } => {
            panic!("Barrier is not supported")
        }
        VariableKind::TensorMap(_) => panic!("Tensor map is not supported"),
    };
    Some(val)
}
