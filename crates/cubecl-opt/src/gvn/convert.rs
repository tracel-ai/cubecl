use std::collections::HashMap;

use cubecl_ir::{ConstantScalarValue, Item, Operation, OperationReflect, Variable, VariableKind};
use float_ord::FloatOrd;
use smallvec::SmallVec;

use super::{Constant, Expression, Local, Value};

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
            Expression::Phi(_) => unreachable!("Phi can't be made into operation"),
        }
    }
}

impl Value {
    pub(crate) fn as_var(&self) -> Variable {
        match self {
            Value::Constant(val) => Variable::constant((*val).into()),
            Value::Local(Local {
                id,
                version: 0,
                item,
            }) => Variable::new(VariableKind::LocalConst { id: *id }, *item),
            Value::Local(Local { id, version, item }) => Variable::new(
                VariableKind::Versioned {
                    id: *id,
                    version: *version,
                },
                *item,
            ),
            Value::Input(id, item) => Variable::new(VariableKind::GlobalInputArray(*id), *item),
            Value::Scalar(id, elem) => {
                Variable::new(VariableKind::GlobalScalar(*id), Item::new(*elem))
            }
            Value::ConstArray(id, item, len) => Variable::new(
                VariableKind::ConstantArray {
                    id: *id,
                    length: *len,
                },
                *item,
            ),
            Value::Builtin(builtin) => Variable::builtin(*builtin),
            Value::Output(id, item) => Variable::new(VariableKind::GlobalOutputArray(*id), *item),
            Value::Slice(id, item) => Variable::new(VariableKind::Slice { id: *id }, *item),
        }
    }
}

pub fn value_of_var(var: &Variable) -> Option<Value> {
    let item = var.item;
    let val = match var.kind {
        VariableKind::GlobalInputArray(id) => Value::Input(id, item),
        VariableKind::GlobalOutputArray(id) => Value::Output(id, item),
        VariableKind::GlobalScalar(id) => Value::Scalar(id, item.elem),
        VariableKind::Versioned { id, version } => Value::Local(Local { id, version, item }),
        VariableKind::LocalConst { id } => Value::Local(Local {
            id,

            version: 0,
            item,
        }),
        VariableKind::ConstantScalar(val) => Value::Constant(val.into()),
        VariableKind::ConstantArray { id, length } => Value::ConstArray(id, item, length),
        VariableKind::LocalMut { .. }
        | VariableKind::SharedMemory { .. }
        | VariableKind::LocalArray { .. }
        | VariableKind::Matrix { .. } => None?,
        VariableKind::Slice { id } => Value::Slice(id, item),
        VariableKind::Builtin(builtin) => Value::Builtin(builtin),
        VariableKind::Pipeline { .. } => panic!("Pipeline is not supported"),
        VariableKind::Barrier { .. } => panic!("Barrier is not supported"),
        VariableKind::TensorMap(_) => panic!("Tensor map is not supported"),
    };
    Some(val)
}

impl From<Constant> for ConstantScalarValue {
    fn from(value: Constant) -> Self {
        match value {
            Constant::Int(val, kind) => ConstantScalarValue::Int(val, kind),
            Constant::Float(val, kind) => ConstantScalarValue::Float(val.0, kind),
            Constant::UInt(val, kind) => ConstantScalarValue::UInt(val, kind),
            Constant::Bool(val) => ConstantScalarValue::Bool(val),
        }
    }
}

impl From<ConstantScalarValue> for Constant {
    fn from(value: ConstantScalarValue) -> Self {
        match value {
            ConstantScalarValue::Int(val, int_kind) => Constant::Int(val, int_kind),
            ConstantScalarValue::Float(val, float_kind) => {
                Constant::Float(FloatOrd(val), float_kind)
            }
            ConstantScalarValue::UInt(val, kind) => Constant::UInt(val, kind),
            ConstantScalarValue::Bool(val) => Constant::Bool(val),
        }
    }
}
