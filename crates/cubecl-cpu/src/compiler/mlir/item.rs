use cubecl_core::ir::Item;
use melior::ir::Type;

use super::prelude::*;

pub(super) trait VisitItem {
    fn visit<'a>(&self, context: &'a Context) -> Type<'a>;
    fn size(&self) -> usize;
}

impl VisitItem for Item {
    fn visit<'a>(&self, context: &'a Context) -> Type<'a> {
        let inner_type = self.elem.visit(context);
        match self.vectorization {
            Some(size) => Type::vector(&[size.get() as u64], inner_type),
            None => inner_type,
        }
    }
    // Get the sizes in bytes
    fn size(&self) -> usize {
        let inner_size = self.elem.size();
        match self.vectorization {
            Some(size) => size.get() as usize * inner_size,
            None => inner_size,
        }
    }
}
