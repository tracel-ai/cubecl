use cubecl_core::ir::{
    Builtin, ConstantScalarValue, FloatKind, IntKind, Item, UIntKind, Variable, VariableKind,
};
use melior::{
    dialect::ods::arith,
    ir::{
        Type, Value,
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::IntegerType,
    },
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
            VariableKind::Builtin(builtin) => self.get_builtin(builtin),
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
                    ConstantScalarValue::UInt(value, int_kind) => {
                        let size = match int_kind {
                            UIntKind::U8 => 8,
                            UIntKind::U16 => 16,
                            UIntKind::U32 => 32,
                            UIntKind::U64 => 64,
                        };
                        let integer_type = IntegerType::new(self.context, size).into();
                        let value = IntegerAttribute::new(integer_type, value as i64).into();
                        arith::constant(self.context, integer_type, value, self.location)
                    }
                    ConstantScalarValue::Float(value, float_kind) => {
                        let float_type = match float_kind {
                            FloatKind::F16 => Type::float16(self.context),
                            FloatKind::BF16 => Type::bfloat16(self.context),
                            FloatKind::F32 => Type::float32(self.context),
                            FloatKind::F64 => Type::float64(self.context),
                            _ => panic!("Type is not supported in LLVM"),
                        };
                        let float_attribute =
                            FloatAttribute::new(self.context, float_type, value).into();
                        arith::constant(self.context, float_type, float_attribute, self.location)
                    }
                    _ => todo!("Operation is not implemented {:?}", constant_scalar_value),
                };
                self.append_operation_with_result(operation)
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
    pub fn get_index(&self, variable: Variable, target_item: Item) -> Value<'a, 'a> {
        let mut index = match variable.kind {
            VariableKind::ConstantScalar(constant_scalar_value) => {
                let value = match constant_scalar_value {
                    ConstantScalarValue::Int(value, _) => value,
                    ConstantScalarValue::UInt(value, _) => value as i64,
                    _ => todo!("Operation is not implemented {}", constant_scalar_value),
                };
                let integer_type = Type::index(self.context);
                let value = IntegerAttribute::new(integer_type, value).into();
                self.append_operation_with_result(arith::constant(
                    self.context,
                    integer_type,
                    value,
                    self.location,
                ))
            }
            VariableKind::Builtin(builtin) => self.get_builtin(builtin),
            _ => todo!("{:?} is not yet implemented", variable.kind),
        };
        if let Some(vectorization) = target_item.vectorization {
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

    pub fn get_builtin(&self, builtin: Builtin) -> Value<'a, 'a> {
        match builtin {
            Builtin::AbsolutePos => self.absolute_pos.unwrap(),
            Builtin::AbsolutePosX => self.absolute_pos_x.unwrap(),
            Builtin::AbsolutePosY => self.absolute_pos_y.unwrap(),
            Builtin::AbsolutePosZ => self.absolute_pos_z.unwrap(),
            Builtin::CubeDimX => self.cube_dim_x.unwrap(),
            Builtin::CubeDimY => self.cube_dim_y.unwrap(),
            Builtin::CubeDimZ => self.cube_dim_z.unwrap(),
            Builtin::CubeCountX => self.cube_count_x.unwrap(),
            Builtin::CubeCountY => self.cube_count_y.unwrap(),
            Builtin::CubeCountZ => self.cube_count_z.unwrap(),
            Builtin::CubePosX => self.cube_count_x.unwrap(),
            Builtin::CubePosY => self.cube_count_y.unwrap(),
            Builtin::CubePosZ => self.cube_count_z.unwrap(),
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
        }
    }
}
