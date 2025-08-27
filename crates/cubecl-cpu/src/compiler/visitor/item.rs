use cubecl_core::ir::{self, ConstantScalarValue, VariableKind};
use tracel_llvm::melior::{
    Context,
    dialect::{arith, ods::vector},
    ir::{
        Attribute, Type, Value,
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::IntegerType,
    },
};

use super::prelude::*;

impl IntoType for ir::Type {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a> {
        let inner_type = self.storage.to_type(context);
        match self.line_size {
            Some(size) if size.get() > 1 => Type::vector(&[size.get() as u64], inner_type),
            _ => inner_type,
        }
    }
    fn is_vectorized(&self) -> bool {
        matches!(self.line_size, Some(size) if size.get() > 1)
    }
}

impl<'a> Visitor<'a> {
    pub fn into_attribute(
        context: &'a Context,
        var: Variable,
        item: ir::Type,
    ) -> Option<Attribute<'a>> {
        let r#type = item.storage.to_type(context);
        match var.kind {
            VariableKind::ConstantScalar(ConstantScalarValue::Float(float, _)) => {
                if item.storage.is_float() {
                    Some(FloatAttribute::new(context, r#type, float).into())
                } else {
                    Some(IntegerAttribute::new(r#type, float as i64).into())
                }
            }
            VariableKind::ConstantScalar(ConstantScalarValue::Bool(bool)) => {
                if item.storage.is_float() {
                    Some(FloatAttribute::new(context, r#type, bool as i64 as f64).into())
                } else {
                    Some(IntegerAttribute::new(r#type, bool as i64).into())
                }
            }
            VariableKind::ConstantScalar(ConstantScalarValue::Int(int, _)) => {
                if item.storage.is_float() {
                    Some(FloatAttribute::new(context, r#type, int as f64).into())
                } else {
                    Some(IntegerAttribute::new(r#type, int).into())
                }
            }
            VariableKind::ConstantScalar(ConstantScalarValue::UInt(u_int, _)) => {
                if item.storage.is_float() {
                    Some(FloatAttribute::new(context, r#type, u_int as f64).into())
                } else {
                    Some(IntegerAttribute::new(r#type, u_int as i64).into())
                }
            }
            _ => None,
        }
    }

    pub fn create_float_constant_from_item(&self, item: ir::Type, constant: f64) -> Value<'a, 'a> {
        let float = item.storage.to_type(self.context);
        let constant = FloatAttribute::new(self.context, float, constant);
        let constant = self.append_operation_with_result(arith::constant(
            self.context,
            constant.into(),
            self.location,
        ));
        let result_type = item.to_type(self.context);
        match item.is_vectorized() {
            true => self.append_operation_with_result(vector::splat(
                self.context,
                result_type,
                constant,
                self.location,
            )),
            false => constant,
        }
    }

    pub fn create_int_constant_from_item(&self, item: ir::Type, constant: i64) -> Value<'a, 'a> {
        let integer = item.storage.to_type(self.context);
        let constant = IntegerAttribute::new(integer, constant);
        let constant = self.append_operation_with_result(arith::constant(
            self.context,
            constant.into(),
            self.location,
        ));
        let result_type = item.to_type(self.context);
        match item.is_vectorized() {
            true => self.append_operation_with_result(vector::splat(
                self.context,
                result_type,
                constant,
                self.location,
            )),
            false => constant,
        }
    }

    pub fn cast_to_bool(&self, value: Value<'a, 'a>, item: ir::Type) -> Value<'a, 'a> {
        let mut bool = IntegerType::new(self.context, 1).into();
        if item.is_vectorized() {
            bool = Type::vector(&[item.line_size.unwrap().get() as u64], bool);
        }
        self.append_operation_with_result(arith::trunci(value, bool, self.location))
    }

    pub fn cast_to_u8(&self, value: Value<'a, 'a>, item: ir::Type) -> Value<'a, 'a> {
        let mut byte = IntegerType::new(self.context, 8).into();
        if item.is_vectorized() {
            byte = Type::vector(&[item.line_size.unwrap().get() as u64], byte);
        }
        self.append_operation_with_result(arith::extui(value, byte, self.location))
    }
}
