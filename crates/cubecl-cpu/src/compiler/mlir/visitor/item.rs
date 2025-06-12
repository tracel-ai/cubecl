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
    // TODO check if vector<1xelem> is better represented as elem alone
    fn to_type<'a>(self, context: &'a Context) -> Type<'a> {
        let inner_type = self.elem.to_type(context);
        match self.vectorization {
            Some(size) => Type::vector(&[size.get() as u64], inner_type),
            _ => inner_type,
        }
    }
}

impl<'a> Visitor<'a> {
    pub fn create_float_constant_from_item(&self, item: Item, constant: f64) -> Value<'a, 'a> {
        let constant = FloatAttribute::new(self.context, Type::float32(self.context), constant);
        let constant = self.append_operation_with_result(arith::constant(
            self.context,
            constant.into(),
            self.location,
        ));
        let vector = item.to_type(self.context);
        self.append_operation_with_result(vector::splat(
            self.context,
            vector,
            constant,
            self.location,
        ))
    }

    pub fn create_int_constant_from_item(&self, item: Item, constant: i64) -> Value<'a, 'a> {
        let integer = item.elem.to_type(self.context);
        let constant = IntegerAttribute::new(integer, constant);
        let constant = self.append_operation_with_result(arith::constant(
            self.context,
            constant.into(),
            self.location,
        ));
        let vector = item.to_type(self.context);
        self.append_operation_with_result(vector::splat(
            self.context,
            vector,
            constant,
            self.location,
        ))
    }
}
