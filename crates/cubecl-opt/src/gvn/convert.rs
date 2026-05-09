use cubecl_ir::{Operation, OperationReflect, Type, Variable, VariableKind};
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
            Expression::Phi(_) => unreachable!("Phi can't be made into operation"),
        }
    }
}

impl Value {
    pub(crate) fn as_var(&self) -> Variable {
        match self {
            Value::Constant(val, ty) => Variable::constant(*val, *ty),
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
            Value::Global(id, item) => Variable::new(VariableKind::GlobalBuffer(*id), *item),
            Value::Scalar(id, elem) => {
                Variable::new(VariableKind::GlobalScalar(*id), Type::new(*elem))
            }
            Value::ConstArray(id, item, len, unroll_factor) => Variable::new(
                VariableKind::ConstantArray {
                    id: *id,
                    length: *len,
                    unroll_factor: *unroll_factor,
                },
                *item,
            ),
            Value::Builtin(builtin, ty) => Variable::builtin(*builtin, *ty),
        }
    }
}

pub fn value_of_var(var: &Variable) -> Option<Value> {
    let item = var.ty;
    let val = match var.kind {
        VariableKind::GlobalBuffer(id) => Value::Global(id, item),
        VariableKind::GlobalScalar(id) => Value::Scalar(id, item.storage_type()),
        VariableKind::Versioned { id, version } => Value::Local(Local { id, version, item }),
        VariableKind::LocalConst { id } => Value::Local(Local {
            id,
            version: 0,
            item,
        }),
        VariableKind::Constant(val) => Value::Constant(val, item),
        VariableKind::ConstantArray {
            id,
            length,
            unroll_factor,
        } => Value::ConstArray(id, item, length, unroll_factor),
        VariableKind::LocalMut { .. }
        | VariableKind::Shared { .. }
        | VariableKind::Matrix { .. } => None?,
        VariableKind::Builtin(builtin) => Value::Builtin(builtin, item.storage_type()),
        VariableKind::Pipeline { .. } => panic!("Pipeline is not supported"),
        VariableKind::BarrierToken { .. } => {
            panic!("Barrier is not supported")
        }
        VariableKind::TensorMap(_) => panic!("Tensor map is not supported"),
        VariableKind::Aggregate { .. } => unreachable!("Should be disaggregated at this point"),
    };
    Some(val)
}
