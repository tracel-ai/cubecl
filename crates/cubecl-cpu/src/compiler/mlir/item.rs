use cubecl_core::ir::Item;
use melior::ir::Type;

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn item_to_type(&self, item: Item) -> Type<'a> {
        let inner_type = self.elem_to_type(item.elem);
        match item.vectorization {
            Some(size) => Type::vector(&[size.get() as u64], inner_type),
            None => inner_type,
        }
    }
}

pub(super) trait ElemSize {
    fn size(self) -> usize;
}

impl ElemSize for Item {
    /// Get the sizes in bytes
    fn size(self) -> usize {
        let inner_size = self.elem.size();
        match self.vectorization {
            Some(size) => size.get() as usize * inner_size,
            None => inner_size,
        }
    }
}
