use cubecl_core::ir::Item;
use melior::{
    dialect::{arith, ods::vector},
    ir::{
        Type, Value,
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::MemRefType,
    },
};

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn create_float_constant_from_item(&self, item: Item, constant: f64) -> Value<'a, 'a> {
        let constant = FloatAttribute::new(self.context, Type::float32(self.context), constant);
        let constant = self.append_operation_with_result(arith::constant(
            self.context,
            constant.into(),
            self.location,
        ));
        let vector = self.item_to_type(item);
        self.append_operation_with_result(vector::splat(
            self.context,
            vector,
            constant,
            self.location,
        ))
    }

    pub fn create_int_constant_from_item(&self, item: Item, constant: i64) -> Value<'a, 'a> {
        let integer = self.elem_to_type(item.elem);
        let constant = IntegerAttribute::new(integer, constant);
        let constant = self.append_operation_with_result(arith::constant(
            self.context,
            constant.into(),
            self.location,
        ));
        let vector = self.item_to_type(item);
        self.append_operation_with_result(vector::splat(
            self.context,
            vector,
            constant,
            self.location,
        ))
    }

    pub fn item_to_type(&self, item: Item) -> Type<'a> {
        let inner_type = self.elem_to_type(item.elem);
        match item.vectorization {
            Some(size) => Type::vector(&[size.get() as u64], inner_type),
            None => inner_type,
        }
    }
    pub fn item_to_memref_buffer_type(&self, item: Item) -> MemRefType<'a> {
        let inner_type = self.elem_to_type(item.elem);
        MemRefType::new(inner_type, &[i64::MIN], None, None)
    }
}
