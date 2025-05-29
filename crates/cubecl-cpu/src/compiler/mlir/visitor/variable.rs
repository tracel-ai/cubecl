use cubecl_core::ir::{ConstantScalarValue, IntKind, Variable, VariableKind};
use melior::{
    dialect::ods::arith,
    ir::{Type, Value, attribute::IntegerAttribute, r#type::IntegerType},
};

use super::Visitor;

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
                        arith::constant(self.context, integer_type, value, self.location)
                    }
                    _ => todo!("Operation is not implemented {}", constant_scalar_value),
                };
                self.append_operation_with_result(operation)
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
    pub fn get_index(&self, variable: Variable) -> Value<'a, 'a> {
        let mut index = match variable.kind {
            VariableKind::ConstantScalar(constant_scalar_value) => {
                let operation = match constant_scalar_value {
                    ConstantScalarValue::Int(value, _) => {
                        let integer_type = Type::index(self.context);
                        let value = IntegerAttribute::new(integer_type, value).into();
                        arith::constant(self.context, integer_type, value, self.location)
                    }
                    _ => todo!("Operation is not implemented {}", constant_scalar_value),
                };
                self.append_operation_with_result(operation)
            }
            VariableKind::Builtin(builtin) => match builtin {
                _ => {
                    let integer_type = Type::index(self.context);
                    let value = IntegerAttribute::new(integer_type, 0).into();
                    self.append_operation_with_result(arith::constant(
                        self.context,
                        integer_type,
                        value,
                        self.location,
                    ))
                }
            },
            _ => todo!("{:?} is not yet implemented", variable.kind),
        };
        if let Some(vectorization) = variable.item.vectorization {
            let vectorization = vectorization.get() as i64;
            let shift = vectorization.ilog2() as i64;
            let constant = self.append_operation_with_result(arith::constant(
                self.context,
                Type::index(self.context),
                IntegerAttribute::new(Type::index(self.context), shift).into(),
                self.location,
            ));
            index = self.append_operation_with_result(arith::shli(
                self.context,
                index,
                constant,
                self.location,
            ));
        }
        index
    }
}
