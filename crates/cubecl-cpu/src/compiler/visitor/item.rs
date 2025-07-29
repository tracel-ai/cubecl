use cubecl_core::ir::{ConstantScalarValue, Item, VariableKind};
use tracel_llvm::melior::{
    Context,
    dialect::{arith, ods::vector},
    ir::{
        Attribute, Type, Value,
        attribute::{FloatAttribute, IntegerAttribute},
    },
};

use super::prelude::*;

impl IntoType for Item {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a> {
        let inner_type = self.elem.to_type(context);
        match self.vectorization {
            Some(size) if size.get() > 1 => Type::vector(&[size.get() as u64], inner_type),
            _ => inner_type,
        }
    }
    fn is_vectorized(&self) -> bool {
        matches!(self.vectorization, Some(size) if size.get() > 1)
    }
}

impl<'a> Visitor<'a> {
    pub fn into_i64(var: Variable) -> i64 {
        match var.kind {
            VariableKind::ConstantScalar(constant_scalar) => constant_scalar.as_i64(),
            _ => panic!("Variable index to access line element is not supported in MLIR"),
        }
    }

    pub fn into_attribute(
        context: &'a Context,
        var: Variable,
        item: Item,
    ) -> Option<Attribute<'a>> {
        let r#type = item.elem.to_type(context);
        match var.kind {
            VariableKind::ConstantScalar(ConstantScalarValue::Float(float, _)) => {
                if item.elem.is_float() {
                    Some(FloatAttribute::new(context, r#type, float).into())
                } else {
                    Some(IntegerAttribute::new(r#type, float as i64).into())
                }
            }
            VariableKind::ConstantScalar(ConstantScalarValue::Bool(bool)) => {
                if item.elem.is_float() {
                    Some(FloatAttribute::new(context, r#type, bool as i64 as f64).into())
                } else {
                    Some(IntegerAttribute::new(r#type, bool as i64).into())
                }
            }
            VariableKind::ConstantScalar(ConstantScalarValue::Int(int, _)) => {
                if item.elem.is_float() {
                    Some(FloatAttribute::new(context, r#type, int as f64).into())
                } else {
                    Some(IntegerAttribute::new(r#type, int).into())
                }
            }
            VariableKind::ConstantScalar(ConstantScalarValue::UInt(u_int, _)) => {
                if item.elem.is_float() {
                    Some(FloatAttribute::new(context, r#type, u_int as f64).into())
                } else {
                    Some(IntegerAttribute::new(r#type, u_int as i64).into())
                }
            }
            _ => None,
        }
    }

    pub fn create_float_constant_from_item(&self, item: Item, constant: f64) -> Value<'a, 'a> {
        let float = item.elem.to_type(self.context);
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

    pub fn create_int_constant_from_item(&self, item: Item, constant: i64) -> Value<'a, 'a> {
        let integer = item.elem.to_type(self.context);
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
}
