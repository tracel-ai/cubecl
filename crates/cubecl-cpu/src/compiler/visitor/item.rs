use cubecl_core::ir::Item;
use tracel_llvm::melior::{
    Context,
    dialect::{arith, ods::vector},
    ir::{
        Type, Value,
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
