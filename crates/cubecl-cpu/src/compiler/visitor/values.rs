use std::collections::HashMap;

use cubecl_core::ir::{self as cube, AddressSpace, Builtin, ConstantValue, FloatKind, ValueKind};
use tracel_llvm::mlir_rs::{
    dialect::{
        index, memref,
        ods::{arith, vector},
    },
    ir::{
        Value,
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::{IntegerType, MemRefType},
    },
};

use super::prelude::*;

pub type Values<'a> = HashMap<ValueKind, Value<'a, 'a>>;

impl<'a> Visitor<'a> {
    pub fn insert_variable(&mut self, variable: cube::Value, value: Value<'a, 'a>) {
        match variable.kind {
            ValueKind::Value { .. } => {
                self.values.insert(variable.kind, value);
            }
            _ => todo!("This variable is not implemented {:?}", variable),
        };
    }

    pub fn declare_mutable_memory(
        &mut self,
        variable: cube::Value,
        value_ty: cube::Type,
        alignment: usize,
    ) {
        let length = value_ty.size() / value_ty.scalar_value_type().size();
        let r#type = value_ty.scalar_value_type().to_type(self.context);
        let align_ty = IntegerType::new(self.context, 64);
        let alignment = IntegerAttribute::new(align_ty.into(), alignment as i64);
        let memref_type = MemRefType::new(r#type, &[length as i64], None, None);
        let value = self
            .first_block
            .unwrap()
            .append_op_result(memref::alloca(
                self.context,
                memref_type,
                &[],
                &[],
                Some(alignment),
                self.location,
            ))
            .unwrap();
        self.values.insert(variable.kind, value);
    }

    pub fn get_binary_op_variable(
        &self,
        lhs: cube::Value,
        rhs: cube::Value,
    ) -> (Value<'a, 'a>, Value<'a, 'a>) {
        let vectorization_factor = std::cmp::max(lhs.vector_size(), rhs.vector_size());
        let (mut lhs_value, mut rhs_value) = (self.get_variable(lhs), self.get_variable(rhs));

        if lhs_value.r#type().is_vector() || rhs_value.r#type().is_vector() {
            if !lhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    lhs.ty.scalar_value_type().to_type(self.context),
                );
                lhs_value = self.append_operation_with_result(vector::broadcast(
                    self.context,
                    vector_type,
                    lhs_value,
                    self.location,
                ));
            }
            if !rhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    rhs.ty.scalar_value_type().to_type(self.context),
                );
                rhs_value = self.append_operation_with_result(vector::broadcast(
                    self.context,
                    vector_type,
                    rhs_value,
                    self.location,
                ));
            }
        }
        (lhs_value, rhs_value)
    }

    pub fn get_memory(&mut self, variable: cube::Value) -> Value<'a, 'a> {
        match variable.kind {
            ValueKind::Value { .. } if let AddressSpace::Global(id) = variable.address_space() => {
                self.args_manager.buffers[id as usize]
            }
            ValueKind::Value { .. } => *self
                .values
                .get(&variable.kind)
                .expect("Value should have been declared before"),
            _ => todo!(
                "This variable isn't backed by memory or implemented: {}",
                variable
            ),
        }
    }

    pub fn get_variable(&self, variable: cube::Value) -> Value<'a, 'a> {
        match variable.kind {
            ValueKind::Value { .. } => *self
                .values
                .get(&variable.kind)
                .expect("Value should have been declared before"),
            ValueKind::Constant(constant_scalar_value) => {
                let (const_type, attribute) = match constant_scalar_value {
                    ConstantValue::Int(value) => {
                        let size = variable.ty.elem_type().size_bits() as u32;

                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute = IntegerAttribute::new(integer_type, value).into();
                        (integer_type, integer_attribute)
                    }
                    ConstantValue::UInt(value) => {
                        let size = variable.ty.elem_type().size_bits() as u32;

                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute =
                            IntegerAttribute::new(integer_type, value as i64).into();
                        (integer_type, integer_attribute)
                    }
                    ConstantValue::Float(value) => {
                        let float_type = match variable.ty.elem_type().as_float().unwrap() {
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
                    ConstantValue::Bool(bool) => {
                        let integer_type = IntegerType::new(self.context, 8).into();
                        let integer_attribute =
                            IntegerAttribute::new(integer_type, bool as i64).into();
                        (integer_type, integer_attribute)
                    }
                };
                let value = self.append_operation_with_result(arith::constant(
                    self.context,
                    const_type,
                    attribute,
                    self.location,
                ));
                match variable.ty.is_vectorized() {
                    true => {
                        let vector = Type::vector(&[variable.vector_size() as u64], const_type);
                        self.append_operation_with_result(vector::broadcast(
                            self.context,
                            vector,
                            value,
                            self.location,
                        ))
                    }
                    false => value,
                }
            }
        }
    }

    pub fn get_index(
        &self,
        variable: cube::Value,
        target_item: cube::Type,
        list_is_vectorized: bool,
    ) -> Value<'a, 'a> {
        let index = self.get_variable(variable);
        let mut index = self.append_operation_with_result(index::casts(
            index,
            Type::index(self.context),
            self.location,
        ));
        if target_item.is_vectorized() && list_is_vectorized {
            let vectorization = target_item.vector_size() as i64;
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
        self.args_manager.get(builtin)
    }
}
