use cubecl_core::ir::Item;
use melior::ir::{Type, r#type::MemRefType};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
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
