use cubecl_core::ir::Item;
use melior::ir::r#type::MemRefType;

use super::prelude::*;

pub(super) trait TransformItem {
    fn visit_type<'a>(&self, context: &'a Context) -> MemRefType<'a>;
}

impl TransformItem for Item {
    fn visit_type<'a>(&self, context: &'a Context) -> MemRefType<'a> {
        let inner_type = self.elem.visit_type(context);
        match self.vectorization {
            Some(size) => MemRefType::new(inner_type, &[i64::MIN, size.get() as i64], None, None),
            None => MemRefType::new(inner_type, &[i64::MIN], None, None),
        }
    }
}
