use cubecl_core::ir::{ConstantScalarValue, IntKind, Variable, VariableKind};
use melior::{
    dialect::ods::arith,
    ir::{BlockLike, Type, Value, attribute::IntegerAttribute, r#type::IntegerType},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn insert_variable(&mut self, variable: Variable, value: Value<'a, 'a>) {
        match variable.kind {
            VariableKind::LocalConst { id } => {
                self.current_local_variables.insert(id, value);
            }
            _ => todo!(),
        };
    }
    pub fn get_variable(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                self.global_buffers[id as usize]
            }
            VariableKind::LocalConst { id } => self
                .current_local_variables
                .get(&id)
                .expect("Variable should have been declared before")
                .clone(),
            VariableKind::Builtin(_builtin) => {
                unreachable!("Builtin should have been removed by a pass before")
            }
            VariableKind::ConstantScalar(constant_scalar_value) => {
                let operation = match constant_scalar_value {
                    ConstantScalarValue::Int(value, int_kind) => {
                        let size = match int_kind {
                            IntKind::I8 => 8,
                            IntKind::I16 => 16,
                            IntKind::I32 => 32,
                            IntKind::I64 => 64,
                        };
                        let integer_type = IntegerType::new(self.context, size).into();
                        let value = IntegerAttribute::new(integer_type, value).into();
                        arith::constant(self.context, integer_type, value, self.location).into()
                    }
                    _ => todo!("Operation is not implemented {}", constant_scalar_value),
                };
                self.block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into()
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
    pub fn get_index(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::ConstantScalar(constant_scalar_value) => {
                let operation = match constant_scalar_value {
                    ConstantScalarValue::Int(value, _) => {
                        let integer_type = Type::index(self.context);
                        let value = IntegerAttribute::new(integer_type, value).into();
                        arith::constant(self.context, integer_type, value, self.location).into()
                    }
                    _ => todo!("Operation is not implemented {}", constant_scalar_value),
                };
                self.block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into()
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
}
