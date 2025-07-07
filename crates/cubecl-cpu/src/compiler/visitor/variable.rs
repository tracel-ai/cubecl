use std::num::NonZero;

use cubecl_core::ir::{
    Builtin, ConstantScalarValue, FloatKind, IntKind, Item, UIntKind, VariableKind,
};
use tracel_llvm::melior::{
    dialect::{
        index, memref,
        ods::{arith, vector},
    },
    ir::{
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::{IntegerType, MemRefType},
    },
};

use super::prelude::*;

impl<'a> Visitor<'a> {
    pub fn insert_variable(&mut self, variable: Variable, value: Value<'a, 'a>) {
        match variable.kind {
            VariableKind::LocalConst { id } => {
                self.current_local_variables.insert(id, value);
            }
            VariableKind::Versioned { id, version } => {
                self.current_version_variables.insert((id, version), value);
            }
            VariableKind::LocalMut { id } => {
                let r#type = variable.elem().to_type(self.context);
                let memref_type = MemRefType::new(
                    r#type,
                    &[variable.vectorization_factor() as i64],
                    None,
                    None,
                );
                let memref = self
                    .current_mut_variables
                    .get(&id)
                    .copied()
                    .unwrap_or_else(|| {
                        let value = self.append_operation_with_result(memref::alloca(
                            self.context,
                            memref_type,
                            &[],
                            &[],
                            None,
                            self.location,
                        ));
                        self.current_mut_variables.insert(id, value);
                        value
                    });
                let integer = IntegerAttribute::new(Type::index(self.context), 0).into();
                let zero = self.append_operation_with_result(arith::constant(
                    self.context,
                    Type::index(self.context),
                    integer,
                    self.location,
                ));
                let operation = if value.r#type().is_vector() {
                    vector::store(self.context, value, memref, &[zero], self.location).into()
                } else {
                    memref::store(value, memref, &[zero], self.location)
                };
                self.block().append_operation(operation);
            }
            _ => todo!("This variable is not implemented {:?}", variable),
        };
    }
    pub fn get_binary_op_variable(
        &self,
        lhs: Variable,
        rhs: Variable,
    ) -> (Value<'a, 'a>, Value<'a, 'a>) {
        let vectorization_factor =
            std::cmp::max(lhs.vectorization_factor(), rhs.vectorization_factor());
        let (mut lhs_value, mut rhs_value) = (self.get_variable(lhs), self.get_variable(rhs));

        if lhs_value.r#type().is_vector() || rhs_value.r#type().is_vector() {
            if !lhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    lhs.elem().to_type(self.context),
                );
                lhs_value = self.append_operation_with_result(vector::splat(
                    self.context,
                    vector_type,
                    lhs_value,
                    self.location,
                ));
            }
            if !rhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    rhs.elem().to_type(self.context),
                );
                rhs_value = self.append_operation_with_result(vector::splat(
                    self.context,
                    vector_type,
                    rhs_value,
                    self.location,
                ));
            }
        }
        (lhs_value, rhs_value)
    }
    pub fn get_memory(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                self.global_buffers[id as usize]
            }
            VariableKind::LocalMut { id } => *self
                .current_mut_variables
                .get(&id)
                .expect("Variable should have been declared before"),
            _ => todo!(
                "This variable isn't backed by memory or implemented: {}",
                variable
            ),
        }
    }
    pub fn is_memory(&self, variable: Variable) -> bool {
        matches!(
            variable.kind,
            VariableKind::GlobalInputArray(_)
                | VariableKind::GlobalOutputArray(_)
                | VariableKind::LocalMut { .. }
        )
    }
    pub fn get_variable(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::LocalConst { id } => *self
                .current_local_variables
                .get(&id)
                .expect("Variable should have been declared before"),
            VariableKind::Builtin(builtin) => self.get_builtin(builtin),
            VariableKind::ConstantScalar(constant_scalar_value) => {
                let (const_type, attribute) = match constant_scalar_value {
                    ConstantScalarValue::Int(value, int_kind) => {
                        let size = match int_kind {
                            IntKind::I8 => 8,
                            IntKind::I16 => 16,
                            IntKind::I32 => 32,
                            IntKind::I64 => 64,
                        };
                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute = IntegerAttribute::new(integer_type, value).into();
                        (integer_type, integer_attribute)
                    }
                    ConstantScalarValue::UInt(value, int_kind) => {
                        let size = match int_kind {
                            UIntKind::U8 => 8,
                            UIntKind::U16 => 16,
                            UIntKind::U32 => 32,
                            UIntKind::U64 => 64,
                        };
                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute =
                            IntegerAttribute::new(integer_type, value as i64).into();
                        (integer_type, integer_attribute)
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
                        (float_type, float_attribute)
                    }
                    _ => todo!("Operation is not implemented {:?}", constant_scalar_value),
                };
                let value = self.append_operation_with_result(arith::constant(
                    self.context,
                    const_type,
                    attribute,
                    self.location,
                ));
                match variable.item.is_vectorized() {
                    true => {
                        let vector =
                            Type::vector(&[variable.vectorization_factor() as u64], const_type);
                        self.append_operation_with_result(vector::splat(
                            self.context,
                            vector,
                            value,
                            self.location,
                        ))
                    }
                    false => value,
                }
            }
            VariableKind::Versioned { id, version } => *self
                .current_version_variables
                .get(&(id, version))
                .expect("Variable should have been declared before"),
            VariableKind::LocalMut { id } => {
                let memref = *self
                    .current_mut_variables
                    .get(&id)
                    .expect("Variable should have been declared before");
                let result_type = variable.item.to_type(self.context);
                let integer = IntegerAttribute::new(Type::index(self.context), 0).into();
                let zero = self.append_operation_with_result(arith::constant(
                    self.context,
                    Type::index(self.context),
                    integer,
                    self.location,
                ));
                if variable.item.is_vectorized() {
                    self.append_operation_with_result(vector::load(
                        self.context,
                        result_type,
                        memref,
                        &[zero],
                        self.location,
                    ))
                } else {
                    self.append_operation_with_result(memref::load(memref, &[zero], self.location))
                }
            }
            VariableKind::GlobalScalar(id) => {
                let var = self.global_scalars[id as usize];
                if variable.item.is_vectorized() {
                    let result_type = variable.item.to_type(self.context);
                    self.append_operation_with_result(vector::load(
                        self.context,
                        result_type,
                        var,
                        &[],
                        self.location,
                    ))
                } else {
                    var
                }
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
    pub fn get_index(&self, variable: Variable, target_item: Item) -> Value<'a, 'a> {
        let index = self.get_variable(variable);
        let mut index = self.append_operation_with_result(index::casts(
            index,
            Type::index(self.context),
            self.location,
        ));
        if target_item.is_vectorized() {
            let vectorization = target_item.vectorization.map(NonZero::get).unwrap_or(1u8) as i64;
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
            Builtin::AbsolutePos => self.absolute_pos,
            Builtin::AbsolutePosX => self.absolute_pos_x,
            Builtin::AbsolutePosY => self.absolute_pos_y,
            Builtin::AbsolutePosZ => self.absolute_pos_z,
            Builtin::CubeDimX => self.cube_dim_x,
            Builtin::CubeDimY => self.cube_dim_y,
            Builtin::CubeDimZ => self.cube_dim_z,
            Builtin::CubeCountX => self.cube_count_x,
            Builtin::CubeCountY => self.cube_count_y,
            Builtin::CubeCountZ => self.cube_count_z,
            Builtin::CubePosX => self.cube_pos_x,
            Builtin::CubePosY => self.cube_pos_y,
            Builtin::CubePosZ => self.cube_pos_z,
            Builtin::UnitPos => self.unit_pos,
            Builtin::UnitPosX => self.unit_pos_x,
            Builtin::UnitPosY => self.unit_pos_y,
            Builtin::UnitPosZ => self.unit_pos_z,
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
